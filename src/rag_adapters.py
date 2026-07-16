import abc
import json
import uuid
import asyncio
import re
from typing import Literal
from qdrant_client import models
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage, SystemMessage
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.graph import MessagesState, START, END, StateGraph
from langgraph.prebuilt import ToolNode
from tenacity import retry, wait_exponential, stop_after_attempt
from lightrag import QueryParam
from src.utils.schemas import GradeDocuments, RewrittenQuestion
from src.utils.documents import load_documents, normalize_text, dedup_preserve_order, truncate_to_token_budget
from src.utils.cost_tracking import embed_query_with_cost, embed_texts_with_cost, calculate_total_cost, message_cost, UsageTrackingCallback, instrument_hipporag
from src.utils.graph_documents import build_global_node_types, sanitize_graph_documents, attach_chunk_nodes, merge_graph_documents
from src.utils.lightrag_support import extract_descriptions_lightrag
from src.config import Config
from langchain_qdrant import QdrantVectorStore


class BaseRAGAdapter(abc.ABC):
    @abc.abstractmethod
    async def index(self, documents: list[str]) -> float:
        pass

    @abc.abstractmethod
    async def retrieve(self, query: str) -> tuple[list[str], float]:
        pass

class VectorRAGAdapter(BaseRAGAdapter):
    def __init__(self, qdrant_client, embedding_service, gemini_client):
        self.client = qdrant_client
        self.gemini_client = gemini_client
        if not self.client.collection_exists(collection_name="vector_storage"):
            self.client.create_collection(
                collection_name="vector_storage",
                vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE)
            )
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name="vector_storage",
            embedding=embedding_service
        )

    async def index(self, documents: list[str]) -> float: # type: ignore
        cost = await asyncio.to_thread(self._sync_index, documents)
        return cost
    
    def _sync_index(self, documents: list[str]) -> float:
        # Clear existing points or recreate collection
        self.client.delete_collection(collection_name="vector_storage")
        self.client.create_collection(
            collection_name="vector_storage",
            vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE)
        )
        # Collapse duplicate chunks before embedding, and assign deterministic
        # content-hash IDs so identical text can never become two distinct points.
        documents = dedup_preserve_order(documents)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, normalize_text(t))) for t in documents]
        # Embed through the litellm gateway (same endpoint as the query path) and
        # upsert precomputed vectors so stored and query vectors stay consistent.
        vectors, cost = embed_texts_with_cost(self.gemini_client, Config.EMBEDDING_MODEL, documents) # type: ignore
        points = [
            models.PointStruct(id=i, vector=v, payload={"page_content": t, "metadata": {}})
            for i, t, v in zip(ids, documents, vectors)
        ]
        self.client.upsert(collection_name="vector_storage", points=points)

        return cost

    @retry(wait=wait_exponential(1, max=10), stop=stop_after_attempt(5))
    async def retrieve(self, query: str) -> tuple[list[str], float]:
        context = await asyncio.to_thread(self._sync_retrieve, query)
        return context

    def _sync_retrieve(self, query: str) -> tuple[list[str], float]:
        # Embed the query through the gateway to capture cost, then search by the
        # precomputed vector. Over-fetch deep enough that the shared token budget,
        # not k, is the binding constraint (median chunk is tiny).
        vector, cost = embed_query_with_cost(self.gemini_client, Config.EMBEDDING_MODEL, query) # type: ignore
        search_result = self.vector_store.similarity_search_by_vector(vector, k=80)
        chunks = dedup_preserve_order([doc.page_content for doc in search_result])
        return truncate_to_token_budget(chunks, Config.MAX_CONTEXT_TOKENS), cost

class LightRAGAdapter(BaseRAGAdapter):
    def __init__(self, rag_instance):
        self.rag = rag_instance

    async def index(self, documents: list[str]):
        # Neo4j is external storage: setup_directories wipes only local dirs, so
        # graph data from earlier runs would merge into this ingest and pollute
        # citation coverage. Drop the workspace first (mirrors Spanner's cleanup).
        drop_status = await self.rag.chunk_entity_relation_graph.drop()
        if drop_status.get("status") != "success":
            raise RuntimeError(f"Neo4j graph wipe failed before indexing: {drop_status}")
        # Hand docling's [source: <file>] marker to LightRAG as file_path so
        # entity/relationship descriptions inherit it; unmarked docs fall back
        # to "unknown_source", which extract_descriptions_lightrag leaves bare.
        matches = (re.match(r"\[source: (.*?)\]", d) for d in documents)
        sources = [m.group(1) if m else "unknown_source" for m in matches]
        await self.rag.ainsert(documents, file_paths=sources)
        cost = calculate_total_cost(Config.TOKEN_TRACKER.get_usage()) # type: ignore
        Config.TOKEN_TRACKER.reset() # type: ignore
        return cost

    @retry(wait=wait_exponential(1, max=10), stop=stop_after_attempt(5))
    async def retrieve(self, query: str) -> tuple[list[str], float]:
        param = QueryParam(
            mode="mix",
            only_need_context=True,
            chunk_top_k=10,
        )
        context = await self.rag.aquery_data(query=query, param=param)
        clean_context = extract_descriptions_lightrag(context) # type: ignore
        cost = calculate_total_cost(Config.TOKEN_TRACKER.get_usage()) # type: ignore
        Config.TOKEN_TRACKER.reset() # type: ignore
        return truncate_to_token_budget(clean_context, Config.MAX_CONTEXT_TOKENS), cost


class HippoRAGAdapter(BaseRAGAdapter):
    def __init__(self, hipporag_instance):
        self.hipporag = hipporag_instance
        # Wraps HippoRAG's internal OpenAI clients so every billable call
        # (OpenIE, rerank, embeddings) is captured; cache hits cost 0.
        self.usage_tracker = instrument_hipporag(hipporag_instance)

    async def index(self, documents: list[str]):
        self.usage_tracker.reset()
        await asyncio.to_thread(self.hipporag.index, docs=documents)
        return self.usage_tracker.cost()

    @retry(wait=wait_exponential(1, max=10), stop=stop_after_attempt(5))
    async def retrieve(self, query: str) -> tuple[list[str], float]:
        if not self.hipporag.fact_embedding_store.embeddings:
            return [], 0.0

        # Reset per attempt so a tenacity re-run doesn't double-count.
        self.usage_tracker.reset()
        results = await asyncio.to_thread(self.hipporag.retrieve, queries=[query])
        context = truncate_to_token_budget(list(results[0].docs), Config.MAX_CONTEXT_TOKENS)

        return context, self.usage_tracker.cost()

class SpannerGraphRAGAdapter(BaseRAGAdapter):
    def __init__(self, graph_store, embedding_service, llm_transformer, graph_name, gemini_client):
        self.graph_store = graph_store
        self.embedding_service = embedding_service
        self.llm_transformer = llm_transformer
        self.graph_name = graph_name
        self.gemini_client = gemini_client
        self._out_edges = None   # {src_label: [non-Chunk edge labels]}, lazy-loaded
        self._node_labels = None

    def _load_edge_schema(self):
        # Cache outgoing non-Chunk edge labels per source label for scoped triple queries.
        ss = self.graph_store.get_structured_schema
        edge_labels = list(ss.get("Edge properties per edge label", {}).keys())
        self._node_labels = list(ss.get("Node properties per node label", {}).keys())
        self._out_edges = {}
        for lbl in edge_labels:
            if lbl.endswith("_Chunk"):
                continue
            src = lbl.split("_", 1)[0]
            self._out_edges.setdefault(src, []).append(lbl)

    def _format_triple(self, node_label, edge_label, node_id, target_id):
        # Strip "{Src}_" prefix and trailing "_{TgtLabel}" to expose the relation type.
        rel = edge_label[len(node_label) + 1:] if edge_label.startswith(node_label + "_") else edge_label
        for nl in (self._node_labels or []):
            if rel.endswith("_" + nl):
                rel = rel[: -(len(nl) + 1)]
                break
        return f"Relationship: {node_id} --{rel}--> {target_id}"

    @retry(wait=wait_exponential(min=4, max=30), stop=stop_after_attempt(15))
    async def _safe_extract_graph(self, doc: Document, config=None):
        # config carries the UsageTrackingCallback so extraction LLM cost is captured.
        res = await self.llm_transformer.aconvert_to_graph_documents([doc], config=config)
        return res[0]

    async def index(self, documents: list[str]) -> float:
        print("Cleaning up graph store before indexing...")
        await asyncio.to_thread(self.graph_store.cleanup)
        
        # Strip the [source: <file>] marker before LLM graph extraction so the
        # extractor doesn't mint filename entities; the Chunk nodes keep the
        # full prefixed text via the zip(documents, ...) pairing below.
        stripped_documents = [re.sub(r"\[source: .*?\]\s*", "", d) for d in documents]
        document_list = load_documents(stripped_documents)
        total_docs = len(document_list)
        
        print(f"Starting Spanner graph extraction for {total_docs} documents...")
        
        # Accumulates token usage from every extraction LLM call (all docs, all
        # retries) via config callback; converted to cost with base-model pricing.
        usage_cb = UsageTrackingCallback()
        extract_config = {"callbacks": [usage_cb]}

        # Concurrency practice: Semaphore to control concurrency levels
        # and gather tasks to execute them concurrently.
        sem = asyncio.Semaphore(5)

        async def extract_with_semaphore(idx, doc):
            async with sem:
                try:
                    print(f"Extracting graph for document {idx+1}/{total_docs}...")
                    graph_doc = await self._safe_extract_graph(doc, extract_config)
                    print(f"Successfully extracted graph for document {idx+1}.")
                    return graph_doc
                except Exception as e:
                    print(f"Failed to extract graph for document {idx+1}: {e}")
                    return None

        tasks = [extract_with_semaphore(i, doc) for i, doc in enumerate(document_list)]
        results = await asyncio.gather(*tasks)
        
        graph_documents_with_chunks = [(text, graph_doc) for text, graph_doc in zip(documents, results) if graph_doc is not None]

        # Validation pipeline (mutates the extracted documents in place, in
        # this order): consistent node types -> node/relationship sanitation
        # -> chunk nodes with MENTIONED_IN links for retrieval.
        global_node_types = build_global_node_types(graph_documents_with_chunks)
        valid_graph_documents = sanitize_graph_documents(graph_documents_with_chunks, global_node_types)
        texts_to_embed, node_references = attach_chunk_nodes(graph_documents_with_chunks)

        embed_cost = 0.0
        if node_references:
            try:
                print(f"Embedding {len(texts_to_embed)} nodes and chunks...")
                # Embed through the gateway so node/chunk vectors match the query path.
                embeddings, embed_cost = await asyncio.to_thread(
                    embed_texts_with_cost, self.gemini_client, Config.EMBEDDING_MODEL, texts_to_embed # type: ignore
                )
                for node, embedding in zip(node_references, embeddings):
                    node.properties["embedding"] = embedding
                print("Successfully embedded nodes.")
            except Exception as e:
                print(f"Failed batched embeddings: {e}")

        if valid_graph_documents:
            merged_doc = merge_graph_documents(valid_graph_documents)

            print("Adding documents to graph store...")
            await asyncio.to_thread(self.graph_store.add_graph_documents, graph_documents=[merged_doc])
            print("Successfully added graph documents to Spanner.")

        # Total indexing cost = LLM graph extraction (all docs + retries) + node/chunk embeddings.
        return usage_cb.cost() + embed_cost

    @retry(wait=wait_exponential(1, max=10), stop=stop_after_attempt(5))
    async def retrieve(self, query: str) -> tuple[list[str], float]:
        query_embeddings, cost = await asyncio.to_thread(
            embed_query_with_cost, self.gemini_client, Config.EMBEDDING_MODEL, query # type: ignore
        )
        query_embeddings_str = ",".join(map(str, query_embeddings))
        
        gql_query = f"""
            GRAPH {self.graph_name}
            MATCH (node)
            WHERE node.embedding IS NOT NULL
            ORDER BY COSINE_DISTANCE(node.embedding, ARRAY[{query_embeddings_str}])
            LIMIT 10
            RETURN SAFE_TO_JSON(node) as node_json
        """
        
        if self._out_edges is None:
            await asyncio.to_thread(self._load_edge_schema)

        responses = await asyncio.to_thread(self.graph_store.query, gql_query)
        chunk_texts = set()
        entity_descriptions = []

        def gql_escape(value: str) -> str:
            # Ids like "Zara'S First Online Store" exist; escape for the string literal.
            return str(value).replace("\\", "\\\\").replace("'", "\\'")

        async def fetch_connected_chunks(node_id, label_str, edge_label):
            chunk_query = f"""
                GRAPH {self.graph_name}
                MATCH (node{label_str})-[e:{edge_label}]-(chunk:Chunk)
                WHERE node.id = '{gql_escape(node_id)}'
                RETURN SAFE_TO_JSON(chunk) as chunk_json
            """
            try:
                chunk_responses = await asyncio.to_thread(self.graph_store.query, chunk_query)
                texts = []
                for chunk_res in chunk_responses:
                    chunk_data = chunk_res["chunk_json"]
                    chunk_el = json.loads(chunk_data.serialize() if hasattr(chunk_data, "serialize") else str(chunk_data))
                    chunk_props = chunk_el.get("properties", {})
                    if "text" in chunk_props:
                        texts.append(chunk_props["text"])
                return texts
            except Exception as e:
                print(f"Failed to fetch connected chunks for node {node_id}: {e}")
                return []

        async def fetch_relationships(node_id, node_label):
            # Outgoing non-Chunk edges as triples; scope by label to stay under the join limit.
            edge_labels = (self._out_edges or {}).get(node_label, [])
            if not edge_labels:
                return []
            alt = "|".join(edge_labels)
            rel_query = f"""
                GRAPH {self.graph_name}
                MATCH (node:{node_label})-[e:{alt}]->()
                WHERE node.id = '{gql_escape(node_id)}'
                RETURN e.target_id AS target_id, SAFE_TO_JSON(e) as e_json
            """
            try:
                rel_responses = await asyncio.to_thread(self.graph_store.query, rel_query)
                triples = []
                for rel_res in rel_responses:
                    e_data = rel_res["e_json"]
                    e_el = json.loads(e_data.serialize() if hasattr(e_data, "serialize") else str(e_data))
                    edge_label = (e_el.get("labels") or [node_label])[0]
                    target_id = rel_res["target_id"]
                    if target_id:
                        triples.append(self._format_triple(node_label, edge_label, node_id, target_id))
                return triples
            except Exception as e:
                print(f"Failed to fetch relationships for node {node_id}: {e}")
                return []

        chunk_tasks, rel_tasks = [], []
        for response in responses:
            try:
                node_data = response["node_json"]
                element = json.loads(node_data.serialize() if hasattr(node_data, "serialize") else str(node_data))

                labels = element.get("labels", [])
                properties = element.get("properties", {})

                if "Chunk" in labels:
                    if "text" in properties:
                        chunk_texts.add(properties["text"])
                    continue

                node_id = properties.get("id") or element.get("id")
                if not node_id:
                    continue

                node_label = labels[0] if labels else "Unknown"
                label_str = f":{node_label}"
                edge_label = f"{node_label}_MENTIONED_IN_Chunk"

                entity_descriptions.append(f"Entity: {node_id} (type: {node_label})")
                chunk_tasks.append(fetch_connected_chunks(node_id, label_str, edge_label))
                rel_tasks.append(fetch_relationships(node_id, node_label))

            except Exception as e:
                print(f"Failed to process top node: {e}")
                continue

        for texts in await asyncio.gather(*chunk_tasks):
            chunk_texts.update(texts)
        relationship_descriptions = []
        for triples in await asyncio.gather(*rel_tasks):
            relationship_descriptions.extend(triples)

        # Descriptions first so the token budget can't evict the graph layer behind huge chunks.
        combined = dedup_preserve_order(entity_descriptions + relationship_descriptions + list(chunk_texts))
        return truncate_to_token_budget(combined, Config.MAX_CONTEXT_TOKENS), cost
    
class AgenticRAGAdapter(BaseRAGAdapter):
    def __init__(self, qdrant_client, embedding_service, gemini_client):
        self.qdrant_client = qdrant_client
        self.gemini_client = gemini_client
        self._query_costs = []
        self._query_rounds = []  # per-round chunk lists, each in similarity order
        if not self.qdrant_client.collection_exists(collection_name="vector_storage"):
            self.qdrant_client.create_collection(
                collection_name="vector_storage",
                vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE)
            )
        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name="vector_storage",
            embedding=embedding_service
        )
        self.responses_dict = {}
        self.response_model = init_chat_model(
            model_provider="openai",
            model=Config.LLM_MODEL_BASE,
            api_key=Config.LLM_BINDING_API_KEY,
            base_url=Config.LLM_BINDING_HOST
        )

        @tool
        async def retrieve_context(query: str) -> str:
            """Search and retrieve available context related to user's query"""
            return await asyncio.to_thread(self._sync_retrieve_context, query)
        
        self.retrieve_context = retrieve_context

        workflow = StateGraph(MessagesState)
        workflow.add_node(self.initial_retrieve)
        workflow.add_node(self.generate_query_or_respond)
        workflow.add_node("_retrieve", ToolNode([self.retrieve_context]))
        workflow.add_node(self.rewrite_question)
        workflow.add_node(self.generate_answer)

        # Force one retrieval up front, then hand off to the grade/rewrite loop.
        workflow.add_edge(START, "initial_retrieve")
        workflow.add_conditional_edges(
            "initial_retrieve",
            self.grade_documents
        )
        workflow.add_conditional_edges(
            "generate_query_or_respond",
            self.route_on_tool_calls,
            {
                "tools": "_retrieve",
                END:"generate_answer"
            }
        )

        workflow.add_conditional_edges(
            "_retrieve",
            self.grade_documents
        )

        workflow.add_edge("generate_answer", END)
        workflow.add_edge("rewrite_question", "generate_query_or_respond")

        self.graph = workflow.compile()
    
    async def index(self, documents: list[str]) -> float:
        return 0.0

    async def retrieve(self, query: str) -> tuple[list[str], float]:
        # Reset per-query embedding cost accumulator before the graph runs; each
        # retrieval round appends its cost in _sync_retrieve_context.
        self._query_costs = []
        self._query_rounds = []

        result = await self.graph.ainvoke({"messages": [SystemMessage(content=Config.RAG_SYSTEM_PROMPT), HumanMessage(content=query)]})


        final_answer = ""

        for message in reversed(result["messages"]):
            if isinstance(message, AIMessage) and message.content:
                final_answer = str(message.content)
                break

        self.responses_dict[query] = final_answer
        return self.return_clean_context(), sum(self._query_costs)


    def get_response(self, query: str) -> str:
        return self.responses_dict.get(query, "")
    

    def return_clean_context(self) -> list[str]:
        # Most-recent round first so the freshest (answer-grounding) chunks survive
        # truncation; within each round keep the original similarity order.
        chunks = []
        for round_chunks in reversed(self._query_rounds):
            for chunk in round_chunks:
                chunks.append(chunk)
        return truncate_to_token_budget(dedup_preserve_order(chunks), Config.MAX_CONTEXT_TOKENS) # type: ignore
    

    def _sync_retrieve_context(self, query: str) -> str:
        # Embed the query through the gateway (capturing cost), then search by the
        # precomputed vector. Over-fetch deep enough per round that the shared token
        # budget, not the per-round cap, is what ultimately binds.
        vector, cost = embed_query_with_cost(self.gemini_client, Config.EMBEDDING_MODEL, query) # type: ignore
        self._query_costs.append(cost)
        search_result = self.vector_store.similarity_search_by_vector(vector, k=80)
        retrieved_context = dedup_preserve_order([doc.page_content for doc in search_result])[:25]
        fused_context = "\n\n".join(retrieved_context)
        self._query_rounds.append(retrieved_context)
        return fused_context

    async def initial_retrieve(self, state: MessagesState):
      
        question = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)][-1].content
        call_id = str(uuid.uuid4())
        fused_context = await asyncio.to_thread(self._sync_retrieve_context, str(question))
        ai_msg = AIMessage(content="", tool_calls=[{"name": "retrieve_context", "args": {"query": str(question)}, "id": call_id}])
        tool_msg = ToolMessage(content=fused_context, tool_call_id=call_id)
        return {"messages": [ai_msg, tool_msg]}

    @retry(wait=wait_exponential(min=4, max=60), stop=stop_after_attempt(10))
    async def generate_query_or_respond(self, state: MessagesState):
        """Call the model to generate a response based on the current state. Given
        the question, it will decide to retrieve using the retriever tool, or simply respond to the user.
        """
        response = await self.response_model.bind_tools([self.retrieve_context]).ainvoke(state["messages"])
        self._query_costs.append(message_cost(response))
        return {"messages": [response]}

    @retry(wait=wait_exponential(min=4, max=60), stop=stop_after_attempt(10))
    async def grade_documents(self, state: MessagesState) -> Literal["generate_answer", "rewrite_question"]:
        """Determine whether the retrieved documents are relevant to the question"""
        question = [msg for msg in state["messages"] if isinstance(msg,
        HumanMessage)][-1].content
        
        context = "\n\n".join(self.return_clean_context()) 
     
        prompt = Config.GRADE_PROMPT.format(context=context, question=question)
        result = await self.response_model.with_structured_output(GradeDocuments, include_raw=True).ainvoke([{"role": "user", "content": prompt}])
        # Capture cost before touching result["parsed"]: a None parse would raise
        # and trigger a tenacity re-run, so appending first avoids double-counting.
        self._query_costs.append(message_cost(result["raw"])) # type: ignore
        response = result["parsed"] # type: ignore

        # Stop condition: prevent infinite loops by limiting max retries
        num_retrievals = sum(1 for msg in state["messages"] if isinstance(msg, ToolMessage))
        
        if response.binary_score == "yes" or response.binary_score == "'yes'" or num_retrievals >= 5: # type: ignore
            return "generate_answer"
        else:
            return "rewrite_question"

    @retry(wait=wait_exponential(min=4, max=60), stop=stop_after_attempt(10))
    async def rewrite_question(self, state: MessagesState):
        """Rewrite the latest question using the context judged not relevant"""

        # [-1] is the most recent question: the original on round 1, otherwise the
        # previous rewrite — so refinement compounds instead of restarting.
        question = [msg for msg in state["messages"] if isinstance(msg,
        HumanMessage)][-1].content

        context = "\n\n".join(self.return_clean_context())
        prompt = Config.REWRITE_PROMPT.format(question=question, context=context)
        result = await self.response_model.with_structured_output(RewrittenQuestion, include_raw=True).ainvoke([{"role": "user", "content": prompt}])
        # Capture cost before touching result["parsed"]: a None parse would raise
        # and trigger a tenacity re-run, so appending first avoids double-counting.
        self._query_costs.append(message_cost(result["raw"])) # type: ignore
        response = result["parsed"] # type: ignore
        return {"messages": [HumanMessage(content=response.rewritten_question)]}


    @retry(wait=wait_exponential(min=4, max=60), stop=stop_after_attempt(10))
    async def generate_answer(self, state: MessagesState):
        """Generate answer to user question and retrieved context"""
        question = [msg for msg in state["messages"] if isinstance(msg,
        HumanMessage)][-1].content
        
        context = "\n\n".join(self.return_clean_context())
        prompt = Config.GENERATE_PROMPT.format(question=question, context=context)
        response = await self.response_model.ainvoke([{"role": "user", "content": prompt}])
        self._query_costs.append(message_cost(response))

        return {"messages": [response]}



    def route_on_tool_calls(self, state: MessagesState):
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "tools"
    
        return END
    
