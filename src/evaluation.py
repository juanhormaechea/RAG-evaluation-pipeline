import os
import asyncio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any
from tenacity import retry, wait_exponential, stop_after_attempt
from ragas.metrics.collections import ContextPrecision, ContextUtilization, ContextRecall, ContextEntityRecall, NoiseSensitivity, AnswerRelevancy, Faithfulness
from ragas.llms import llm_factory
from src.config import Config
from src.utils import JudgeGradingScheme, calculate_average_score, calculate_final_score

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



def _initialize_dataframes(strategies: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    scorer_df = pd.read_csv("./templates/CrossDoc_RAG_Scoring_Template_v2.csv")
    scoring_summary_df = pd.read_csv("./templates/CrossDoc_RAG_Scoring_Summary.csv")

    PREFIX_MAP = {
        "vector_rag": "vector",
        "lightrag": "lightrag",
        "hipporag": "hipporag",
        "spanner_graph": "graph",
        "agentic_rag": "agentic"
    }

    for strategy in strategies:
        prefix = PREFIX_MAP[strategy]
        for measure in ["correctness", "nugget_recall", "faithful", "retrieval", "attribution", "latency_ms", "cost_usd"]:
            col_name = f"{prefix}_{measure}"
            if col_name not in scorer_df.columns:
                scorer_df[col_name] = pd.Series(dtype="float64")
    return scorer_df, scoring_summary_df


async def _run_llm_evaluations(
    df: pd.DataFrame,
    retrieval_df: pd.DataFrame,
    openai_client,
    strategies: list[str],
    scorer_df: pd.DataFrame,
    is_aggregated: bool
) -> tuple[list[Any], list[tuple[str, int, str]]]:
    PREFIX_MAP = {
        "vector_rag": "vector",
        "lightrag": "lightrag",
        "hipporag": "hipporag",
        "spanner_graph": "graph",
        "agentic_rag": "agentic"
    }
    
    sem = asyncio.Semaphore(10)

    @retry(wait=wait_exponential(min=4, max=60), stop=stop_after_attempt(10))
    async def safe_score(prompt):
        async with sem:
            return await openai_client.responses.parse(
                model=Config.LLM_MODEL,
                input=[
                    {"role": "system", "content": Config.JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                text_format=JudgeGradingScheme
            )

    tasks = []
    task_metadata = []  

    for strategy in strategies:
        prefix = PREFIX_MAP[strategy]
        for i in range(df.shape[0]):
            if is_aggregated:
                scorer_df.at[i, f"{prefix}_latency_ms"] = retrieval_df.at["retrieval_time", strategy]
            else:
                scorer_df.at[i, f"{prefix}_latency_ms"] = retrieval_df[strategy].iloc[i] * 1000.0 # type:ignore
            
            strategy_dataset = df.at[i, f"dataset_{strategy}"]
            prompt = Config.USER_PROMPT.format(**strategy_dataset) # type: ignore
            tasks.append(safe_score(prompt))
            task_metadata.append((strategy, i, strategy_dataset["query_type"])) # type: ignore

    responses = await asyncio.gather(*tasks)
    return responses, task_metadata


def _parse_and_store_metrics(
    scorer_df: pd.DataFrame,
    responses: list[Any],
    task_metadata: list[tuple[str, int, str]]
) -> pd.DataFrame:
    PREFIX_MAP = {
        "vector_rag": "vector",
        "lightrag": "lightrag",
        "hipporag": "hipporag",
        "spanner_graph": "graph",
        "agentic_rag": "agentic"
    }
    for res, (strategy, i, query_type) in zip(responses, task_metadata):
        output = res.output_parsed.model_dump()
        prefix = PREFIX_MAP[strategy]
        
        for metric in ["correctness", "nugget_recall", "faithful", "retrieval", "attribution"]:
            scorer_df.at[i, f"{prefix}_{metric}"] = output.get(metric)
    return scorer_df


def _calculate_final_scores(scorer_df: pd.DataFrame, strategies: list[str]) -> pd.DataFrame:
    PREFIX_MAP = {
        "vector_rag": "vector",
        "lightrag": "lightrag",
        "hipporag": "hipporag",
        "spanner_graph": "graph",
        "agentic_rag": "agentic"
    }
    for strategy in strategies:
        prefix = PREFIX_MAP[strategy]
        scorer_df[f"{prefix}_final_score"] = 0.0
        for i in range(scorer_df.shape[0]):
            q_type = scorer_df.at[i, "query_type"]
            use_recall = q_type in ["multi_doc_entity", "global_thematic"]
            
            final_score = calculate_final_score( # type: ignore
                faithfulness=scorer_df.at[i, f"{prefix}_faithful"], # type:ignore
                correctness=None if use_recall else scorer_df.at[i, f"{prefix}_correctness"], # type:ignore
                nugget_recall=scorer_df.at[i, f"{prefix}_nugget_recall"], # type:ignore
                retrieval=scorer_df.at[i, f"{prefix}_retrieval"], # type:ignore
                attribution=scorer_df.at[i, f"{prefix}_attribution"], # type:ignore
                unanswerable=(q_type == "unanswerable")
            )
            scorer_df.at[i, f"{prefix}_final_score"] = final_score
    return scorer_df


def _aggregate_summary(scorer_df: pd.DataFrame, scoring_summary_df: pd.DataFrame) -> pd.DataFrame:
    grouped_means = scorer_df.groupby("query_type").mean(numeric_only=True)

    for i in range(scoring_summary_df.shape[0]):
        system = scoring_summary_df.at[i, "system"]
        q_type = scoring_summary_df.at[i, "query_type"]
        
        if system == "vector":
            score_cols, lat_cols, faith_cols = ["vector_final_score"], ["vector_latency_ms"], ["vector_faithful"]
        elif system == "agentic":
            score_cols, lat_cols, faith_cols = ["agentic_final_score"], ["agentic_latency_ms"], ["agentic_faithful"]
        elif system == "graph":
            score_cols = ["lightrag_final_score", "hipporag_final_score", "graph_final_score"]
            lat_cols = ["lightrag_latency_ms", "hipporag_latency_ms", "graph_latency_ms"]
            faith_cols = ["lightrag_faithful", "hipporag_faithful", "graph_faithful"]
        else:
            continue
            
        if q_type == "ALL":
            scoring_summary_df.at[i, "avg_score"] = scorer_df[score_cols].mean().mean()
            scoring_summary_df.at[i, "avg_latency_ms"] = scorer_df[lat_cols].mean().mean()
            scoring_summary_df.at[i, "faithfulness_rate"] = scorer_df[faith_cols].mean().mean()
        else:
            scoring_summary_df.at[i, "avg_score"] = grouped_means.loc[q_type, score_cols].mean() # type: ignore
            scoring_summary_df.at[i, "avg_latency_ms"] = grouped_means.loc[q_type, lat_cols].mean() # type: ignore
            scoring_summary_df.at[i, "faithfulness_rate"] = grouped_means.loc[q_type, faith_cols].mean() # type: ignore
    return scoring_summary_df


def _generate_visualizations(scorer_df: pd.DataFrame, scoring_summary_df: pd.DataFrame) -> None:
    grouped_means = scorer_df.groupby("query_type").mean(numeric_only=True)
    query_types = [
        "cross_doc_synthesis", "consistency_check", "multi_doc_entity", 
        "cross_doc_multi_hop", "global_thematic", "factoid_single_doc", 
        "unanswerable", "ALL"
    ]
    
    fig, axes = plt.subplots(8, 2, figsize=(16, 32))
    for (ax_main, ax_breakdown), q_type in zip(axes, query_types):
        subset = scoring_summary_df[scoring_summary_df["query_type"] == q_type]
        sns.barplot(data=subset, x="system", y="avg_score", ax=ax_main)
        ax_main.set_title(q_type.replace("_", " ").title())
        ax_main.set(xlabel="RAG solutions", ylabel="Score")
        ax_main.set_ylim(0.0, 1.0)

        if q_type == "ALL":
            breakdown_data = {
                "system": ["lightrag", "hipporag", "spanner_graph"],
                "avg_score": [
                    scorer_df["lightrag_final_score"].mean(),
                    scorer_df["hipporag_final_score"].mean(),
                    scorer_df["graph_final_score"].mean()  
                ]
            }
        else:
            breakdown_data = {
                "system": ["lightrag", "hipporag", "spanner_graph"],
                "avg_score": [
                    grouped_means.loc[q_type, "lightrag_final_score"],
                    grouped_means.loc[q_type, "hipporag_final_score"],
                    grouped_means.loc[q_type, "graph_final_score"] 
                ]
            }
        
        breakdown_df = pd.DataFrame(breakdown_data)
        sns.barplot(data=breakdown_df, x="system", y="avg_score", ax=ax_breakdown, palette="Set2")
        ax_breakdown.set_title(f"{q_type.replace('_', ' ').title()} - Graph breakdown")
        ax_breakdown.set(xlabel="Graph RAG solutions", ylabel="Score")
        ax_breakdown.set_ylim(0.0, 1.0)

    plt.tight_layout()
    plt.show()


async def evaluate_retrieval(df: pd.DataFrame, retrieval_df: pd.DataFrame, openai_client, strategies: list[str]) -> pd.DataFrame:
    scorer_df, scoring_summary_df = _initialize_dataframes(strategies)
    is_aggregated = "retrieval_time" in retrieval_df.index

    responses, task_metadata = await _run_llm_evaluations(
        df, retrieval_df, openai_client, strategies, scorer_df, is_aggregated
    )

    scorer_df = _parse_and_store_metrics(scorer_df, responses, task_metadata)
    scorer_df = _calculate_final_scores(scorer_df, strategies)
    scoring_summary_df = _aggregate_summary(scorer_df, scoring_summary_df)

    scorer_df.to_csv("./templates/CrossDoc_RAG_Scoring_Template_v2.csv", index=False)
    scoring_summary_df.to_csv("./templates/CrossDoc_RAG_Scoring_Summary.csv", index=False)

    _generate_visualizations(scorer_df, scoring_summary_df)
    
    average_latency = {}
    for strategy in strategies:
        if is_aggregated:
            average_latency[strategy] = retrieval_df.at["retrieval_time", strategy]
        else:
            average_latency[strategy] = retrieval_df[strategy].mean() * 1000.0

    return pd.DataFrame(average_latency, index=["retrieval_time"])
