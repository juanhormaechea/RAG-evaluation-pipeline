"""Pricing and usage tracking for LLM and embedding API calls."""
import logging
import threading
from langchain_core.callbacks import AsyncCallbackHandler


def calculate_total_cost(usage_data: dict) -> float:
    # gemini 2.5 flash pricing, change at discretion
    INPUT_PRICE_PER_MILLION = 0.3
    OUTPUT_PRICE_PER_MILLION = 2.5

    prompt_cost = (usage_data["prompt_tokens"] / 1000000) * INPUT_PRICE_PER_MILLION
    completion_cost = (usage_data["completion_tokens"] / 1000000) * OUTPUT_PRICE_PER_MILLION

    return prompt_cost + completion_cost


def message_cost(message) -> float:
    """USD cost of a LangChain AIMessage, priced at base-model (gemini-2.5-flash) rates.

    Prefers the normalized usage_metadata; falls back to the provider-raw
    response_metadata['token_usage'] so a gateway that omits usage_metadata does
    not silently record 0.0.
    """
    usage = getattr(message, "usage_metadata", None)
    if usage:
        return calculate_total_cost({
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
        })
    token_usage = (getattr(message, "response_metadata", None) or {}).get("token_usage")
    if token_usage:
        return calculate_total_cost({
            "prompt_tokens": token_usage.get("prompt_tokens", 0),
            "completion_tokens": token_usage.get("completion_tokens", 0),
        })
    logging.warning("AIMessage has no token usage; recording cost as 0.0")
    return 0.0


def embed_query_with_cost(client, model: str, text: str) -> tuple[list[float], float]:
    # Raw response gives access to the litellm gateway cost header alongside the vector.
    raw = client.embeddings.with_raw_response.create(model=model, input=text)
    cost = float(raw.headers.get("x-litellm-response-cost") or 0.0)
    embedding = raw.parse().data[0].embedding
    return embedding, cost


def embed_texts_with_cost(client, model: str, texts: list[str],
                          batch_size: int = 250) -> tuple[list[list[float]], float]:
    vectors, total_cost = [], 0.0
    for i in range(0, len(texts), batch_size):
        raw = client.embeddings.with_raw_response.create(model=model, input=texts[i:i + batch_size])
        total_cost += float(raw.headers.get("x-litellm-response-cost") or 0.0)
        # sort by .index to guarantee input order
        vectors.extend(d.embedding for d in sorted(raw.parse().data, key=lambda d: d.index))
    return vectors, total_cost


class UsageTrackingCallback(AsyncCallbackHandler):
    """Accumulate prompt/completion tokens from every LLM call in a run.

    Pass an instance via `config={"callbacks": [handler]}` to capture usage from
    calls hidden inside LangChain runnables (e.g. LLMGraphTransformer). Async
    on_llm_end runs inline on the event loop, so concurrent runs can't race on the
    counters (no lock needed). Prefers the normalized usage_metadata, falling back
    to the provider-raw response_metadata['token_usage'] like message_cost.
    """

    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.llm_ends = 0  # billed LLM calls seen (incl. internal retries)

    async def on_llm_end(self, response, **kwargs):
        self.llm_ends += 1
        for gens in response.generations:
            for gen in gens:
                message = getattr(gen, "message", None)
                usage = getattr(message, "usage_metadata", None)
                if usage:
                    self.prompt_tokens += usage.get("input_tokens", 0)
                    self.completion_tokens += usage.get("output_tokens", 0)
                    continue
                token_usage = (getattr(message, "response_metadata", None) or {}).get("token_usage")
                if token_usage:
                    self.prompt_tokens += token_usage.get("prompt_tokens", 0)
                    self.completion_tokens += token_usage.get("completion_tokens", 0)

    def cost(self) -> float:
        return calculate_total_cost({
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        })


class HippoRAGUsageTracker:
    """Thread-safe accumulator for HippoRAG API usage.

    HippoRAG's OpenIE stage fires LLM calls from a ThreadPoolExecutor, so
    unlike UsageTrackingCallback (event-loop serialized) the counters need a
    lock. LLM tokens are priced with calculate_total_cost (base-model rates);
    embedding cost comes from the litellm gateway response-cost header —
    the same split SpannerGraphRAG uses.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        with self._lock:
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.embedding_cost = 0.0

    def add_llm_usage(self, prompt_tokens: int, completion_tokens: int):
        with self._lock:
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens

    def add_embedding_cost(self, cost: float):
        with self._lock:
            self.embedding_cost += cost

    def cost(self) -> float:
        with self._lock:
            return calculate_total_cost({
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
            }) + self.embedding_cost


def instrument_hipporag(hipporag_instance) -> HippoRAGUsageTracker:
    """Capture HippoRAG API usage by wrapping its OpenAI clients in place.

    HippoRAG returns per-call token metadata internally but discards it, so
    the only complete caller-side hook is the two sync openai clients every
    billable call goes through: llm_model.openai_client (OpenIE NER/triple
    extraction and DSPy fact reranking — the rerank filter binds llm_model.infer
    at construction, so wrapping infer would miss it) and
    embedding_model.client (chunk/entity/fact stores and query embedding).
    Calls served from HippoRAG's response cache never reach the client and
    correctly count as $0. No HippoRAG source modification required.
    """
    tracker = HippoRAGUsageTracker()

    completions = hipporag_instance.llm_model.openai_client.chat.completions
    original_chat_create = completions.create

    def tracked_chat_create(*args, **kwargs):
        response = original_chat_create(*args, **kwargs)
        usage = getattr(response, "usage", None)
        if usage:
            tracker.add_llm_usage(
                getattr(usage, "prompt_tokens", 0) or 0,
                getattr(usage, "completion_tokens", 0) or 0,
            )
        return response

    completions.create = tracked_chat_create

    embeddings = hipporag_instance.embedding_model.client.embeddings
    # Capture the raw-response variant before patching .create: with_raw_response
    # re-reads .create on each access, so capturing it later would recurse.
    original_raw_embed_create = embeddings.with_raw_response.create

    def tracked_embed_create(*args, **kwargs):
        # Raw response exposes the litellm gateway cost header alongside the data.
        raw = original_raw_embed_create(*args, **kwargs)
        tracker.add_embedding_cost(float(raw.headers.get("x-litellm-response-cost") or 0.0))
        return raw.parse()

    embeddings.create = tracked_embed_create

    return tracker
