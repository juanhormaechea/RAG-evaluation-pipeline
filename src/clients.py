from dataclasses import dataclass
from qdrant_client import QdrantClient
from openai import OpenAI, AsyncOpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_spanner import SpannerGraphStore
from langchain_experimental.graph_transformers import LLMGraphTransformer
from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings
from src.config import Config

@dataclass
class AppClients:
    qdrant_client: QdrantClient
    openai_client: OpenAI
    async_openai_client: AsyncOpenAI
    llm: ChatOpenAI
    embedding_service: OpenAIEmbeddings
    llm_transformer: LLMGraphTransformer
    ragas_embeddings: RagasOpenAIEmbeddings
    graph_store: SpannerGraphStore

def get_clients() -> AppClients:
    Config.setup_env()
    
    import httpx
    
    qdrant_client = QdrantClient(path=Config.QDRANT_STORAGE_DIR)
    openai_client = OpenAI(api_key=Config.LLM_BINDING_API_KEY, base_url=Config.LLM_BINDING_HOST)
    
    # Use a robust async http client to prevent Connection Errors during heavy Ragas evaluation
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=500, max_keepalive_connections=100),
        timeout=httpx.Timeout(120.0, connect=60.0)
    )
    async_openai_client = AsyncOpenAI(api_key=Config.LLM_BINDING_API_KEY, base_url=Config.LLM_BINDING_HOST, http_client=http_client)
    
    llm = ChatOpenAI(
        # model="gpt-4o-mini-2024-07-18",
        # api_key=Config.OPENAI_API_KEY # type: ignore
        model=Config.LLM_MODEL, # type: ignore
        api_key=Config.LLM_BINDING_API_KEY, # type: ignore
        base_url=Config.LLM_BINDING_HOST
    )
    
    embedding_service = OpenAIEmbeddings(
        # model="text-embedding-3-small",
        # dimensions=1536,
        # api_key=Config.OPENAI_API_KEY, # type: ignore
        model=Config.EMBEDDING_MODEL, # type: ignore
        api_key=Config.EMBEDDING_BINDING_API_KEY, # type: ignore
        base_url=Config.EMBEDDING_BINDING_HOST, 
        chunk_size=250
    )
    
    llm_transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=["Person", "Organization", "Location", "Concept", "Product", "Technology", "Process", "Metric", "Event", "Unknown"],
        allowed_relationships=["RELATED_TO", "USES", "PRODUCES", "AFFECTS", "PART_OF", "LOCATED_IN", "IMPLEMENTS", "ACHIEVES"]
    )
    
    ragas_embeddings = RagasOpenAIEmbeddings(async_openai_client, model=Config.EMBEDDING_MODEL)# type: ignore
    
    graph_store = SpannerGraphStore(
        instance_id=Config.INSTANCE_ID,
        database_id=Config.DATABASE_ID,
        graph_name=Config.GRAPH_NAME
    )
    
    return AppClients(
        qdrant_client=qdrant_client,
        openai_client=openai_client,
        async_openai_client=async_openai_client,
        llm=llm,
        embedding_service=embedding_service,
        llm_transformer=llm_transformer,
        ragas_embeddings=ragas_embeddings,
        graph_store=graph_store
    )
