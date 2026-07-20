"""LightRAG bindings: model/embedding functions, initialization, context parsing."""
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag import LightRAG
from src.config import Config
from src.utils.documents import dedup_preserve_order
from typing import Any


async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    return await openai_complete_if_cache(
        model=Config.LLM_MODEL_BASE, # type: ignore
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        # api_key=Config.OPENAI_API_KEY,
        api_key=Config.LLM_BINDING_API_KEY,
        base_url=Config.LLM_BINDING_HOST,
        token_tracker=Config.TOKEN_TRACKER,
        **kwargs
    )


async def embedding_func(texts: list[str]):
    return await openai_embed.func(
        texts,
        # model="text-embedding-3-small",
        # api_key=Config.OPENAI_API_KEY
        model=Config.EMBEDDING_MODEL,
        api_key=Config.EMBEDDING_BINDING_API_KEY,
        base_url=Config.EMBEDDING_BINDING_HOST,
        token_tracker=Config.TOKEN_TRACKER
    )


async def initialize_lightrag(working_dir: str = Config.WORKING_DIR):
    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(embedding_dim=768, func=embedding_func),
        graph_storage="Neo4JStorage"
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag


def extract_descriptions_lightrag(raw_context: dict[str, Any]) -> list[str]:
    if raw_context.get("status") != "success":
        return []

    data = raw_context.get("data", {})
    contexts: list[str] = []

    def _source_prefix(file_path: str | None) -> str:
        if file_path and file_path != "unknown_source":
            return f"[source: {file_path}] "
        return ""

    # Document chunks: raw source text, most reliable for faithfulness/attribution.
    for chunk in data.get("chunks", []):
        content = (chunk.get("content") or "").strip()
        if not content:
            continue
        contexts.append(f"{_source_prefix(chunk.get('file_path'))}{content}")

    # Relationship descriptions: keep the src->tgt endpoints so the synthesized
    # fact stays interpretable (dropping them was the old parser's information loss).
    for rel in data.get("relationships", []):
        desc = (rel.get("description") or "").strip()
        if not desc:
            continue
        src_id, tgt_id = rel.get("src_id", "?"), rel.get("tgt_id", "?")
        contexts.append(f"{_source_prefix(rel.get('file_path'))}Relationship ({src_id} - {tgt_id}): {desc}")

    # Entity descriptions: named, typed graph nodes relevant to the query.
    for ent in data.get("entities", []):
        desc = (ent.get("description") or "").strip()
        if not desc:
            continue
        name = ent.get("entity_name", "?")
        etype = ent.get("entity_type") or "UNKNOWN"
        contexts.append(f"{_source_prefix(ent.get('file_path'))}Entity {name} ({etype}): {desc}")

    return dedup_preserve_order(contexts)
