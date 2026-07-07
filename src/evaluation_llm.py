import asyncio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tenacity import retry, wait_exponential, stop_after_attempt
from typing import Any
from src.config import Config
from src.utils import JudgeGradingScheme, JudgeReasoningGradingScheme


def _initialize_llm_data(strategies: list[str]) -> tuple[pd.DataFrame, dict[str, list[Any]], dict[str, list[Any]]]:
    scorer_df = pd.read_csv("./templates/CrossDoc_RAG_Scoring_Template_v2.csv")
    results: dict[str, list[Any]] = { 
        "vector_rag": [],
        "lightrag": [],
        "hipporag": [],
        "spanner_graph": [],
        "agentic_rag": []
    }

    averages: dict[str, list[Any]] = {
        "vector_rag": [0] * 5,
        "lightrag": [0] * 5,
        "hipporag": [0] * 5,
        "spanner_graph": [0] * 5,
        "agentic_rag": [0] * 5
    }

    # Pre-initialize dynamically any strategy-specific columns to avoid alignment issues in pandas
    for strategy in strategies:
        prefix = strategy.replace("spanner_graph", "graph").replace("_rag", "")
        for measure in ["correctness", "nugget_recall", "faithful", "retrieval", "attribution"]:
            col_name = f"{prefix}_{measure}"
            if col_name not in scorer_df.columns:
                scorer_df[col_name] = pd.Series(dtype="float64")

    return scorer_df, results, averages


async def _run_llm_strategy_evaluations(
    df: pd.DataFrame,
    openai_client,
    strategy: str
) -> list[Any]:
    sem = asyncio.Semaphore(10)

    @retry(wait=wait_exponential(min=4, max=60), stop=stop_after_attempt(10))
    async def safe_score(prompt):
        async with sem:
            return await openai_client.responses.parse(
                model="gpt-4o-mini-2024-07-18",
                input=[
                    {"role": "system", "content": Config.JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                text_format=JudgeGradingScheme
            )

    tasks = []
    for i in range(df.shape[0]):
        strategy_dataset = df.at[i, f"dataset_{strategy}"]
        prompt = Config.USER_PROMPT.format(**strategy_dataset) # type: ignore
        tasks.append(safe_score(prompt))

    return await asyncio.gather(*tasks)


def _parse_and_store_llm_metrics(
    scorer_df: pd.DataFrame,
    responses: list[Any],
    results: dict[str, list[Any]],
    averages: dict[str, list[Any]],
    strategy: str,
    num_rows: int
) -> None:
    for res in responses:
        results[strategy].append(res.output_parsed)
    
    prefix = strategy.replace("spanner_graph", "graph").replace("_rag", "")
    for i in range(len(results[strategy])):
        parsed_dict = results[strategy][i].model_dump()
        for a, measure in enumerate(["correctness", "nugget_recall", "faithful", "retrieval", "attribution"]):
            val = parsed_dict[measure]
            averages[strategy][a] += val
            column = f"{prefix}_{measure}"
            scorer_df.at[i, column] = val

    averages[strategy] = np.array(averages[strategy]) / num_rows # type: ignore


def _generate_llm_visualizations(averages: dict[str, list[Any]]) -> None:
    metric_names = ["correctness", "nugget_recall", "faithful", "retrieval", "attribution"]
    average_df = pd.DataFrame(averages, index=metric_names).T

    fig, axes = plt.subplots(1, 3, figsize=(14, 7))
    sns.barplot(x=average_df.index, y=average_df["correctness"], ax=axes[0])
    axes[0].set_title("Correctness")
    
    sns.barplot(x=average_df.index, y=average_df["nugget_recall"], ax=axes[1])
    axes[1].set_title("Nugget Recall")

    sns.barplot(x=average_df.index, y=average_df["faithful"], ax=axes[2])
    axes[2].set_title("Faithfulness")

    plt.tight_layout()
    plt.show()

    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 7))
    sns.barplot(x=average_df.index, y=average_df["retrieval"], ax=axes2[0])
    axes2[0].set_title("Retrieval")

    sns.barplot(x=average_df.index, y=average_df["attribution"], ax=axes2[1])
    axes2[1].set_title("Attribution")

    plt.tight_layout()
    plt.show()


async def evaluate_retrieval(df: pd.DataFrame, openai_client, strategies: list[str]):
    scorer_df, results, averages = _initialize_llm_data(strategies)

    for strategy in strategies:
        responses = await _run_llm_strategy_evaluations(df, openai_client, strategy)
        _parse_and_store_llm_metrics(scorer_df, responses, results, averages, strategy, df.shape[0])

    scorer_df.to_csv("./templates/CrossDoc_RAG_Scoring_Template_v2.csv", index=False)
    _generate_llm_visualizations(averages)


    
        

    

    


