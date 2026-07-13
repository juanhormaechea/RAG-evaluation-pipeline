"""LightRAG bindings: model/embedding functions, initialization, context parsing."""
import json
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag import LightRAG
from src.config import Config


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


async def initialize_lightrag():
    rag = LightRAG(
        working_dir=Config.WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(embedding_dim=768, func=embedding_func),
        graph_storage="Neo4JStorage"
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag


def extract_descriptions_lightrag(raw_context: str) -> list[str]:
    chunks = []
    graph_elements = []

    for line in raw_context.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("```") or any(x in line for x in ["Knowledge Graph Data", "Document Chunks", "Reference Document List"]):
            continue
        if line.startswith("{") and line.endswith("}"):
            try:
                data = json.loads(line)
                if "content" in data:
                    # Text chunk
                    chunks.append(data["content"])
                elif "description" in data:
                    # Entity or relationship description
                    desc = data["description"].split("<SEP>")[0]
                    graph_elements.append(desc)
                else:
                    graph_elements.append(line)
            except json.JSONDecodeError:
                graph_elements.append(line)
        else:
            graph_elements.append(line)

    # Combine chunks and group the scattered graph metadata into a single string block
    contexts = list(chunks)
    if graph_elements:
        contexts.append("Retrieved Graph Entities and Relationships:\n" + "\n".join(graph_elements))

    return contexts
