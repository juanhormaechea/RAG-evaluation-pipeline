# RAG Benchmarking and Evaluation Pipeline Report

---

## Testing Pipeline Overview

### 1. Initialization and Setup

- **VectorRAG**: Traditional embedding-based retrieval.
- **LightRAG**: A lightweight graph-based retrieval approach.
- **HippoRAG**: An advanced graph RAG solution.
- **SpannerGraphRAG**: A comprehensive graph RAG using Google Cloud Spanner.
- **AgenticRAG**: An agent-driven approach capable of autonomous reasoning during retrieval.

A standardized benchmarking dataset is loaded to track queries, ground truths, and evaluation results.

### 2. Document Indexing

The target documents (in this case, PowerPoint presentations) are parsed, chunked, and processed. These chunks are then fed into each RAG adapter to build their respective knowledge bases (indexes).

- **Metric Captured**: **Indexing Interval** (the total time required by each system to process and index the corpus).

### 3. Context Retrieval

For each query in the benchmark dataset, the pipeline queries every RAG system to retrieve the most relevant context.

- **Metric Captured**: **Retrieval Interval** (the latency for each system to search and return relevant information).

### 4. LLM Response Generation

Using the retrieved contexts from the previous step, a standard prompt is generated and sent to the LLM to synthesize a final answer.
*(Note: AgenticRAG generates its own response organically as part of its retrieval loop, so it bypasses this standard LLM completion step).*

### 5. Evaluation Phase

The results are compiled into an evaluation dataset, which maps the original query, ground truth, retrieved context, and the system's generated answer. An LLM-as-a-judge approach is utilized to evaluate each response against predefined metrics.

### 6. Analysis and Visualization

Finally, the pipeline aggregates the scores and performance data (latency/intervals) to generate comparative visualizations. This enables an intuitive understanding of trade-offs between speed, cost, and answer quality across the evaluated architectures.

---

## Evaluation Metrics

The evaluation phase relies on an LLM judge to grade the responses based on a strict set of criteria. The metrics assessed are:

- **Correctness**: Measures the fraction of the reference's true atomic facts that are accurately conveyed by the system's final answer.
- **Nugget Recall**: Specifically useful for enumeration or list-based answers, this measures how many of the expected items were covered, while penalizing the model for hallucinating spurious extras.
- **Faithfulness**: Assesses whether the claims made in the system's answer are strictly entailed by the retrieved context, penalizing information drawn from the LLM's intrinsic knowledge (hallucination).
- **Retrieval**: A binary or continuous check on whether the retrieved context actually contained the necessary source documents or information required to answer the query.
- **Attribution**: Evaluates whether each sentence in the final answer correctly cites the supporting source document.
- **Latency & Cost**: Tracks the operational efficiency of the system, calculating the time taken to retrieve/index and the estimated API cost.

---

## Hypothesis and Query Types

Based on the benchmark template, specific RAG methodologies are hypothesized to perform better depending on the nature of the query. The hypotheses map to the following query types:

| Query Type                    | Description                                                                                            | Favored RAG Method            | Rationale                                                                                                                        |
| :---------------------------- | :----------------------------------------------------------------------------------------------------- | :---------------------------- | :------------------------------------------------------------------------------------------------------------------------------- |
| **Factoid Single-Doc**  | Simple, direct queries that rely on a single fact from a specific document.                            | **Vector RAG**          | Standard semantic search is highly efficient and accurate for direct fact-retrieval without the overhead of graph traversals.    |
| **Consistency Check**   | Queries requiring validation of facts or checking for contradictions across documents.                 | **Agentic RAG**         | Agents can perform iterative fact-checking and reasoning loops to identify inconsistencies.                                      |
| **Multi-Doc Entity**    | Queries concerning a specific entity whose information is scattered across multiple documents.         | **Graph RAG**           | Graph structures naturally link entities across different source documents via relationships.                                    |
| **Global Thematic**     | Broad queries asking for overarching themes or summaries of the entire corpus.                         | **Graph RAG**           | Graphs can abstract high-level communities and themes using community detection algorithms.                                      |
| **Cross-Doc Multi-Hop** | Complex queries requiring the system to connect a chain of facts across different documents.           | **Agentic RAG**         | Agents can reason step-by-step, retrieving one fact to inform the search query for the next fact in the chain.                   |
| **Cross-Doc Synthesis** | Queries requiring the amalgamation of disparate concepts from multiple sources into a cohesive answer. | **Graph / Agentic RAG** | Both methods excel here; graphs provide the structural links, while agents provide the reasoning to synthesize the final answer. |

---

## HippoRAG Cost Measurement

HippoRAG (2.0.0a3) extracts per-call token metadata internally but never exposes it: `CacheOpenAI.infer` returns `(message, metadata, cache_hit)`, yet OpenIE accumulates the counts only in local scope and the DSPy rerank filter discards them, while `OpenAIEmbeddingModel.encode` drops the response `usage` entirely. With LiteLLM admin endpoints (`/key/info`, `/user/info`) unavailable, costs are measured caller-side via `instrument_hipporag()` (`src/utils.py`), which wraps — in place, no source modification — the two sync OpenAI clients every billable call flows through:

- `llm_model.openai_client.chat.completions.create` → token counts priced at base-model rates via `calculate_total_cost` (covers OpenIE NER/triple extraction and DSPy fact reranking; wrapping `llm_model.infer` instead would miss reranking, since the filter binds that method at construction).
- `embedding_model.client.embeddings.create` → rerouted through `with_raw_response` to read the gateway's `x-litellm-response-cost` header (covers chunk/entity/fact stores, synonymy edges, and query embeddings), matching how the other adapters price embeddings.

`HippoRAGAdapter` resets the tracker before each `index`/`retrieve` and reports the accumulated cost, aligning HippoRAG with the cost accounting of the other systems.

Known limitations:

- Calls served from HippoRAG's SQLite LLM cache and its embedding cache never reach the wrapped clients and therefore report $0 — which is correct, as no API spend occurs on re-runs.
- If the gateway omits the `x-litellm-response-cost` header, that embedding call contributes 0.0 rather than an estimate.
- A retrieval attempt that fails and is retried only reports the final attempt's spend (same semantics as the other adapters).
