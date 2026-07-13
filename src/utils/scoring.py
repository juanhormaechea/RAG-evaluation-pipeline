"""Final/average score formulas for the LLM-judge evaluation metrics."""
from typing import Any


def calculate_final_score(faithfulness: float, correctness: float | None, nugget_recall: float | None, retrieval: float, attribution: float, unanswerable: bool) -> float:
    if unanswerable:
        return faithfulness

    if correctness is None and nugget_recall is None:
        raise ValueError("must provide a value for either correctness or nugget recall")

    primary_metric = nugget_recall if correctness is None else correctness

    s_final = (0.3 * faithfulness) + (0.3 * primary_metric) + (0.2 * retrieval) + (0.2 * attribution) # type: ignore

    return s_final


def calculate_average_score(dataset_list: list[dict[str, Any]], query_type: str) -> float:
    if not dataset_list:
        return 0.0

    final_score_sum = 0.0

    for dataset in dataset_list:
        unanswerable = (query_type == "unanswerable")
        use_recall = query_type in ["multi_doc_entity", "global_thematic"]
        correctness_val = None if use_recall else dataset.get("correctness")

        final_score = calculate_final_score(
            faithfulness=dataset["faithful"],
            correctness=correctness_val,
            nugget_recall=dataset.get("nugget_recall"),
            retrieval=dataset["retrieval"],
            attribution=dataset["attribution"],
            unanswerable=unanswerable
        )
        final_score_sum += final_score

    return (final_score_sum / len(dataset_list))
