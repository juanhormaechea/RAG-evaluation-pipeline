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

Finally, the pipeline aggregates the scores and performance data (latency/intervals) to generate comparative visualizations. This enables an intuitive understanding of trade-offs between speed, cost, and answer quality across the evaluated architectures. When several chunking strategies are evaluated (see below), their results are further consolidated into a single cross-strategy comparison.

---

## Chunking Strategy Comparison

How a corpus is split into indexed units, that is, **chunk granularity**, strongly affects every RAG method. Fine chunks favour precise, targeted retrieval, while coarser chunks pack more context (and more source documents) into each retrieved unit. To make this trade-off measurable, the harness runs the **entire pipeline** (index → retrieve → generate → evaluate) once per chunking strategy and compares the outcomes side by side.

### Building the strategies

`process_pptx_file` produces fine-grained chunks straight from the documents using docling. This is the **baseline**. From there, `fuse_strings(contents, min_tokens)` sorts the chunks by length and greedily merges the smallest ones together until each fused chunk crosses a token threshold, yielding progressively coarser corpora. Three thresholds give three additional strategies:

| Strategy      | Content list                     | Description                                 |
| :------------ | :------------------------------- | :------------------------------------------ |
| `baseline`  | `contents`                     | Raw docling chunks, unmodified (finest).    |
| `fuse_1000` | `fuse_strings(contents, 1000)` | Small chunks fused to a ~1,000-token floor. |
| `fuse_2000` | `fuse_strings(contents, 2000)` | Fused to a ~2,000-token floor.              |
| `fuse_4000` | `fuse_strings(contents, 4000)` | Fused to a ~4,000-token floor (coarsest).   |

### Isolated per-strategy runs

Each strategy is executed by `run_pipeline`, which builds a fresh set of adapters with **their own storage**  `./outputs/{strategy}` for HippoRAG and `./data/rag_storage/{strategy}` for LightRAG and writes its scores to a dedicated `./results/{strategy}/` directory. HippoRAG and LightRAG persist their indexes on disk and would otherwise accumulate chunks from earlier strategies, letting one run's corpus leak into the next and distort the comparison.

### Comparing the results

Once every strategy has run, `compare_runs` collects their summaries into a single **RAG method × chunking strategy** view covering final score, quality score, latency, retrieval + generation cost, and indexing cost. The combined table is saved to `./results/comparison.csv`, with an accompanying grouped bar chart at `./results/comparison.png`.

### Resuming after a crash

If a run fails partway through, simply re-run the pipeline cell — finished strategies are reused and only the unfinished work is redone. Do **not** re-run the setup cell (`Config.setup_directories()`) first: it starts a clean benchmark and discards everything computed so far.

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
