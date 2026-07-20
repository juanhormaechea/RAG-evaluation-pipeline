import os
import asyncio
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any
from tenacity import retry, wait_exponential, stop_after_attempt
from src.config import Config
from src.utils.schemas import JudgeGradingScheme
from src.utils.scoring import calculate_final_score

PREFIX_MAP = {
    "vector_rag": "vector",
    "lightrag": "lightrag",
    "hipporag": "hipporag",
    "spanner_graph": "graph",
    "agentic_rag": "agentic"
}

QUERY_TYPES = [
        "cross_doc_synthesis", "consistency_check", "multi_doc_entity", 
        "cross_doc_multi_hop", "global_thematic", "factoid_single_doc", 
        "unanswerable", "ALL"
    ]

def add_eval_dataset(df: pd.DataFrame, strategies: list[str]) -> None:
    for strategy in strategies:
        datasets = []
        for i in range(df.shape[0]):
            datasets.append({
                "query": str(df.at[i, "query"]),
                "query_type": str(df.at[i, "query_type"]),
                "reference_ground_truth": str(df.at[i, "ground_truth"]),
                "expected_source_documents": str(df.at[i, "source_documents"]),
                "retrieved_context": df.at[i, f"results_{strategy}"],
                "system_answer": str(df.at[i, f"actual_responses_{strategy}"])
            }) 
        df[f"dataset_{strategy}"] = datasets  



def _initialize_dataframes(strategies: list[str], df_benchmark_data: pd.DataFrame) -> pd.DataFrame:
    # Build the scorer from the benchmark rows actually being evaluated (not a
    # re-read of the full CSV) so the positional writes in _run_llm_evaluations
    # stay aligned when a subset like short_benchmark.csv is in play.
    base_cols = [c for c in ["query_id", "query_type", "num_docs_required", "query",
                             "ground_truth", "source_documents", "hypothesis_favors"]
                 if c in df_benchmark_data.columns]
    scorer_df = df_benchmark_data[base_cols].copy().reset_index(drop=True)

    for strategy in strategies:
        prefix = PREFIX_MAP[strategy]
        for measure in ["correctness", "nugget_recall", "faithful", "retrieval", "attribution", "latency_ms", "cost_usd"]:
            col_name = f"{prefix}_{measure}"
            if col_name not in scorer_df.columns:
                scorer_df[col_name] = pd.Series(dtype="float64")
    return scorer_df


async def _run_llm_evaluations(
    df: pd.DataFrame,
    e2e_df: pd.DataFrame,
    e2e_cost_df: pd.DataFrame,
    gemini_client,
    openai_client,
    strategies: list[str],
    scorer_df: pd.DataFrame,
    is_aggregated: bool
) -> tuple[list[Any], list[tuple[str, int, str]]]:
    
    sem = asyncio.Semaphore(10)

    @retry(wait=wait_exponential(min=4, max=60), stop=stop_after_attempt(10))
    async def safe_score(client, prompt: str, model: str):
        async with sem:
            return await client.responses.parse(
                model=model,
                input=[
                    {"role": "system", "content": Config.JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                text_format=JudgeGradingScheme
            )

    tasks_model_1 = []
    tasks_model_2 = []
    tasks_model_3 = []
    task_metadata = []  

    for strategy in strategies:
        prefix = PREFIX_MAP[strategy]
        for i in range(df.shape[0]):
            if is_aggregated:
                scorer_df.at[i, f"{prefix}_latency_ms"] = e2e_df.at["retrieval_time", strategy]
            else:
                scorer_df.at[i, f"{prefix}_latency_ms"] = e2e_df[strategy].iloc[i] * 1000.0 # type:ignore
            
            scorer_df.at[i, f"{prefix}_cost_usd"] = e2e_cost_df[strategy].iloc[i]

            strategy_dataset = df.at[i, f"dataset_{strategy}"]
            prompt = Config.USER_PROMPT.format(**strategy_dataset) # type: ignore
            tasks_model_1.append(safe_score(openai_client, prompt, Config.LLM_MODEL_1))
            tasks_model_2.append(safe_score(gemini_client, prompt, Config.LLM_MODEL_2))
            tasks_model_3.append(safe_score(gemini_client, prompt, Config.LLM_MODEL_3))
            task_metadata.append((strategy, i, strategy_dataset["query_type"])) # type: ignore



    responses_model_1, responses_model_2, responses_model_3 = await asyncio.gather(
        asyncio.gather(*tasks_model_1),
        asyncio.gather(*tasks_model_2),
        asyncio.gather(*tasks_model_3)
    )


    return [responses_model_1, responses_model_2, responses_model_3], task_metadata


def _parse_and_store_metrics(
    scorer_df: pd.DataFrame,
    response_list: list[Any],
    task_metadata: list[tuple[str, int, str]]
) -> pd.DataFrame:
    
    metrics = ["correctness", "nugget_recall", "faithful", "retrieval", "attribution"]

    df1 = pd.DataFrame([res.output_parsed.model_dump() for res in response_list[0]])
    df2 = pd.DataFrame([res.output_parsed.model_dump() for res in response_list[1]])
    df3 = pd.DataFrame([res.output_parsed.model_dump() for res in response_list[2]])

    avg_df = (df1[metrics] + df2[metrics] + df3[metrics]) / 3.0

    for idx, (strategy, i, query_type) in enumerate(task_metadata):
        prefix = PREFIX_MAP[strategy]
        for metric in metrics:
            scorer_df.at[i, f"{prefix}_{metric}"] = avg_df.at[idx, metric]
        

    return scorer_df


def _calculate_final_scores(scorer_df: pd.DataFrame, strategies: list[str]) -> pd.DataFrame:

    for strategy in strategies:
        prefix = PREFIX_MAP[strategy]
        scorer_df[f"{prefix}_final_score"] = 0.0
        scorer_df[f"{prefix}_quality_score"] = 0.0
        for i in range(scorer_df.shape[0]):
            q_type = scorer_df.at[i, "query_type"]
            use_recall = q_type in ["multi_doc_entity", "global_thematic"]
            faithful = scorer_df.at[i, f"{prefix}_faithful"]
            primary = scorer_df.at[i, f"{prefix}_nugget_recall"] if use_recall else scorer_df.at[i, f"{prefix}_correctness"]

            final_score = calculate_final_score( # type: ignore
                faithfulness=faithful, # type:ignore
                correctness=None if use_recall else scorer_df.at[i, f"{prefix}_correctness"], # type:ignore
                nugget_recall=scorer_df.at[i, f"{prefix}_nugget_recall"], # type:ignore
                retrieval=scorer_df.at[i, f"{prefix}_retrieval"], # type:ignore
                attribution=scorer_df.at[i, f"{prefix}_attribution"], # type:ignore
                unanswerable=(q_type == "unanswerable")
            )
            scorer_df.at[i, f"{prefix}_final_score"] = final_score
            # Citation-independent fallback: rank on answer quality alone if the
            # [source:] marker pipeline leaves retrieval/attribution unmeasurable.
            quality = faithful if q_type == "unanswerable" else 0.5 * faithful + 0.5 * primary # type: ignore
            scorer_df.at[i, f"{prefix}_quality_score"] = quality
    return scorer_df


def _aggregate_summary(scorer_df: pd.DataFrame) -> pd.DataFrame:
    """Build the per-(query_type, system) summary from scratch.

    One row per individual system — the spanner adapter is reported as
    "spanner_graph", never folded into a combined "graph" average with
    lightrag/hipporag. n_queries comes from the scored rows themselves.
    """
    grouped_means = scorer_df.groupby("query_type").mean(numeric_only=True)
    type_counts = scorer_df["query_type"].value_counts()

    rows = []
    for q_type in QUERY_TYPES:
        if q_type == "ALL":
            stats, n = scorer_df.mean(numeric_only=True), int(scorer_df.shape[0])
        elif q_type in grouped_means.index:
            stats, n = grouped_means.loc[q_type], int(type_counts[q_type])
        else:
            continue  # type absent from this benchmark run (e.g. smoke tests)
        for system, prefix in PREFIX_MAP.items():
            rows.append({
                "benchmark_set": "cross_doc",
                "query_type": q_type,
                "n_queries": n,
                "system": system,
                "avg_score": stats[f"{prefix}_final_score"],
                "quality_score": stats[f"{prefix}_quality_score"],
                "faithfulness_rate": stats[f"{prefix}_faithful"],
                "avg_latency_ms": stats[f"{prefix}_latency_ms"],
                "avg_cost_usd": stats[f"{prefix}_cost_usd"],
            })
    return pd.DataFrame(rows)


def _generate_visualizations(scoring_summary_df: pd.DataFrame, results_dir: str = "./results", show: bool = True) -> None:
    fig, axes = plt.subplots(len(QUERY_TYPES), 1, figsize=(12, 4 * len(QUERY_TYPES)))
    for ax, q_type in zip(axes, QUERY_TYPES):
        subset = scoring_summary_df[scoring_summary_df["query_type"] == q_type]
        if subset.empty:
            ax.set_visible(False)
            continue
        sns.barplot(data=subset, x="system", y="avg_score", ax=ax)
        ax.set_title(f"{q_type.replace('_', ' ').title()} (n={int(subset['n_queries'].iloc[0])})")
        ax.set(xlabel="RAG solutions", ylabel="Score")
        ax.set_ylim(0.0, 1.0)

    plt.tight_layout()
    # Persist per-run so a multi-strategy loop keeps every figure instead of
    # clobbering a single inline render.
    os.makedirs(results_dir, exist_ok=True)
    fig.savefig(os.path.join(results_dir, "scores.png"), dpi=100, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)  # batch runs save to disk; don't render or leak figures


async def evaluate_retrieval(df: pd.DataFrame, e2e_df: pd.DataFrame, e2e_cost_df: pd.DataFrame, gemini_client, openai_client, strategies: list[str], results_dir: str = "./results", show: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    scorer_df = _initialize_dataframes(strategies, df)
    is_aggregated = "retrieval_time" in e2e_df.index

    response_list, task_metadata = await _run_llm_evaluations(
        df, e2e_df, e2e_cost_df, gemini_client, openai_client, strategies, scorer_df, is_aggregated
    )

    scorer_df = _parse_and_store_metrics(scorer_df, response_list, task_metadata)
    scorer_df = _calculate_final_scores(scorer_df, strategies)
    scoring_summary_df = _aggregate_summary(scorer_df)

    os.makedirs(results_dir, exist_ok=True)
    scorer_df.to_csv(os.path.join(results_dir, "scores.csv"), index=False)
    scoring_summary_df.to_csv(os.path.join(results_dir, "scoring_summary.csv"), index=False)

    _generate_visualizations(scoring_summary_df, results_dir, show)
    
    average_latency = {}
    average_cost = {}
    for strategy in strategies:
        if is_aggregated:
            average_latency[strategy] = e2e_df.at["retrieval_time", strategy]
        else:
            average_latency[strategy] = e2e_df[strategy].mean() * 1000.0
        
        average_cost[strategy] = e2e_cost_df[strategy].mean()

    return pd.DataFrame(average_latency, index=["retrieval_time"]), pd.DataFrame(average_cost, index=["retrieval_cost"])
