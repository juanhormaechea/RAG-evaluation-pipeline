import os
import asyncio
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import openai
from hipporag import HippoRAG
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from src.config import Config
from src.rag_adapters import (
    VectorRAGAdapter,
    LightRAGAdapter,
    HippoRAGAdapter,
    SpannerGraphRAGAdapter,
    AgenticRAGAdapter,
)
from src.utils.benchmark import initialize_dataframe, generate_prompt
from src.utils.lightrag_support import initialize_lightrag
from src.evaluation import add_eval_dataset, evaluate_retrieval


async def build_adapters(clients, label: str) -> dict:
    hipporag = HippoRAG(
        save_dir=f"./outputs/{label}",
        llm_model_name=Config.LLM_MODEL_BASE,
        llm_base_url=Config.LLM_BINDING_HOST,
        embedding_model_name=Config.EMBEDDING_MODEL,
        embedding_base_url=Config.EMBEDDING_BINDING_HOST,
    )
    lightrag = await initialize_lightrag(working_dir=f"./data/rag_storage/{label}")

    return {
        "vector_rag": VectorRAGAdapter(clients.qdrant_client, clients.embedding_service, clients.gemini_client),
        "lightrag": LightRAGAdapter(lightrag),
        "hipporag": HippoRAGAdapter(hipporag),
        "spanner_graph": SpannerGraphRAGAdapter(
            clients.graph_store,
            clients.embedding_service,
            clients.llm_transformer,
            Config.GRAPH_NAME,
            clients.gemini_client,
        ),
        "agentic_rag": AgenticRAGAdapter(clients.qdrant_client, clients.embedding_service, clients.gemini_client),
    }


@retry(wait=wait_exponential(min=4, max=60), stop=stop_after_attempt(10),
       retry=retry_if_exception_type(openai.RateLimitError))
async def _safe_completion(client, prompt: str):
    return await client.chat.completions.with_raw_response.create(
        model=Config.LLM_MODEL_BASE,
        messages=[{"role": "user", "content": prompt}],
    )


async def _timed_completion(client, prompt: str) -> tuple[str, float, float]:
    start = datetime.datetime.now()
    r = await _safe_completion(client, prompt)
    elapsed = (datetime.datetime.now() - start).total_seconds()
    cost = float(r.headers.get("x-litellm-response-cost") or 0.0)
    answer = r.parse().choices[0].message.content
    return answer, elapsed, cost


async def run_pipeline(
    content_list: list[str],
    label: str,
    clients,
    benchmark_csv: str,
    results_root: str = "./results",
) -> dict:

    adapters = await build_adapters(clients, label)
    strategies = list(adapters.keys())
    df = initialize_dataframe(benchmark_csv, strategies)

    # --- Indexing ---
    indexing_costs = {name: 0.0 for name in strategies}
    for name, adapter in adapters.items():
        indexing_costs[name] = await adapter.index(content_list)
    # Agentic shares vector_rag's collection (its own index() is a no-op).
    indexing_costs["agentic_rag"] = indexing_costs["vector_rag"]
    indexing_costs_df = pd.DataFrame(indexing_costs, index=["indexing_cost"])

    # --- Retrieval ---
    retrieval_intervals: dict[str, list[float]] = {name: [] for name in strategies}
    retrieval_costs: dict[str, list[float]] = {name: [] for name in strategies}
    for i in range(df.shape[0]):
        query = str(df.at[i, "query"])
        await asyncio.sleep(1)
        for name, adapter in adapters.items():
            start = datetime.datetime.now()
            context, costs = await adapter.retrieve(query)
            retrieval_intervals[name].append((datetime.datetime.now() - start).total_seconds())
            retrieval_costs[name].append(costs)
            df.at[i, f"results_{name}"] = context

    retrieval_interval_df = pd.DataFrame(retrieval_intervals, index=list(df["query_id"]))
    retrieval_cost_df = pd.DataFrame(retrieval_costs, index=list(df["query_id"]))

    # --- Generation ---
    generation_intervals: dict[str, list[float]] = {name: [] for name in strategies}
    generation_costs: dict[str, list[float]] = {name: [] for name in strategies}
    async_gemini_client = clients.async_gemini_client
    for i in range(df.shape[0]):
        query = str(df.at[i, "query"])
        tasks, names = [], []
        await asyncio.sleep(1)
        for name in strategies:
            # Agentic generated its answer during its retrieval loop; its
            # generation cost/latency is already folded into retrieval.
            if name == "agentic_rag":
                df.at[i, f"actual_responses_{name}"] = adapters[name].get_response(query)
                generation_intervals[name].append(0.0)
                generation_costs[name].append(0.0)
                continue
            context = df.at[i, f"results_{name}"]
            prompt = generate_prompt(query, context)  # type: ignore
            names.append(name)
            tasks.append(_timed_completion(async_gemini_client, prompt))

        responses = await asyncio.gather(*tasks)
        for name, (answer, elapsed, cost) in zip(names, responses):
            df.at[i, f"actual_responses_{name}"] = answer
            generation_intervals[name].append(elapsed)
            generation_costs[name].append(cost)

    generation_interval_df = pd.DataFrame(generation_intervals, index=list(df["query_id"]))
    generation_cost_df = pd.DataFrame(generation_costs, index=list(df["query_id"]))

    # --- Evaluation ---
    e2e_retrieval_df = retrieval_interval_df + generation_interval_df
    e2e_cost_df = retrieval_cost_df + generation_cost_df
    add_eval_dataset(df, strategies)

    results_dir = os.path.join(results_root, label)
    e2e_latency_df, e2e_cost_summary_df = await evaluate_retrieval(
        df,
        e2e_retrieval_df,
        e2e_cost_df,
        clients.async_gemini_client,
        clients.async_openai_client,
        strategies,
        results_dir=results_dir,
        show=False,  # per-strategy figures saved to disk; keep the batch cell quiet
    )

    return {
        "label": label,
        "df": df,
        "indexing_costs_df": indexing_costs_df,
        "e2e_latency_df": e2e_latency_df,
        "e2e_cost_df": e2e_cost_summary_df,
        "scoring_summary_df": pd.read_csv(os.path.join(results_dir, "scoring_summary.csv")),
        "scores_df": pd.read_csv(os.path.join(results_dir, "scores.csv")),
    }


def compare_runs(all_runs: dict[str, dict], results_root: str = "./results") -> pd.DataFrame:
    labels = list(all_runs.keys())

    frames = []
    for label, res in all_runs.items():
        summary = res["scoring_summary_df"].copy()
        summary["chunk_strategy"] = label
        frames.append(summary)
    combined = pd.concat(frames, ignore_index=True)
    overall = combined[combined["query_type"] == "ALL"].copy()

    # Indexing cost lives outside the scoring summary; splice it in per (strategy, system).
    idx_rows = []
    for label, res in all_runs.items():
        idx = res["indexing_costs_df"]  # index ["indexing_cost"], columns = systems
        for system in idx.columns:
            idx_rows.append({
                "chunk_strategy": label,
                "system": system,
                "indexing_cost_usd": float(idx.at["indexing_cost", system]),
            })
    overall = overall.merge(pd.DataFrame(idx_rows), on=["chunk_strategy", "system"], how="left")

    os.makedirs(results_root, exist_ok=True)
    overall.to_csv(os.path.join(results_root, "comparison.csv"), index=False)

    metrics = [
        ("avg_score", "Final Score"),
        ("quality_score", "Quality Score"),
        ("avg_latency_ms", "Latency (ms)"),
        ("avg_cost_usd", "Retrieval+Gen Cost (USD)"),
        ("indexing_cost_usd", "Indexing Cost (USD)"),
    ]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 5 * len(metrics)))
    for ax, (col, title) in zip(axes, metrics):
        pivot = overall.pivot(index="system", columns="chunk_strategy", values=col)
        pivot = pivot.reindex(columns=labels)  # keep strategy order, not alphabetical
        pivot.plot(kind="bar", ax=ax)
        ax.set_title(f"{title} by RAG method and chunking strategy")
        ax.set(xlabel="RAG solution", ylabel=title)
        ax.legend(title="chunk strategy")

    plt.tight_layout()
    fig.savefig(os.path.join(results_root, "comparison.png"), dpi=100, bbox_inches="tight")
    plt.show()

    return overall
