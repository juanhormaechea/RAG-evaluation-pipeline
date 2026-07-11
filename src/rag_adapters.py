import abc
import json
import uuid
import asyncio
import hashlib
from typing import Literal
from qdrant_client import models
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage, SystemMessage
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langgraph.graph import MessagesState, START, END, StateGraph
from langgraph.prebuilt import ToolNode
from tenacity import retry, wait_exponential, stop_after_attempt
from lightrag import QueryParam
from src.utils import extract_descriptions_lightrag, GradeDocuments, RewrittenQuestion, load_documents, normalize_text, dedup_preserve_order, embed_query_with_cost, embed_texts_with_cost, calculate_total_cost, message_cost, UsageTrackingCallback
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
        # precomputed vector. Over-fetch, then trim to 5 distinct chunks so residual
        # duplicate points can't fill the result set.
        vector, cost = embed_query_with_cost(self.gemini_client, Config.EMBEDDING_MODEL, query) # type: ignore
        search_result = self.vector_store.similarity_search_by_vector(vector, k=20)
        return dedup_preserve_order([doc.page_content for doc in search_result])[:5], cost

class LightRAGAdapter(BaseRAGAdapter):
    def __init__(self, rag_instance):
        self.rag = rag_instance

    async def index(self, documents: list[str]):
         # type: ignore
        await self.rag.ainsert(documents)
        cost = calculate_total_cost(Config.TOKEN_TRACKER.get_usage()) # type: ignore
        Config.TOKEN_TRACKER.reset() # type: ignore
        return cost

    @retry(wait=wait_exponential(1, max=10), stop=stop_after_attempt(5))
    async def retrieve(self, query: str) -> tuple[list[str], float]:
        param = QueryParam(
            mode="mix",
            only_need_context=False,
            enable_rerank=True,
            top_k=5,
            chunk_top_k=5,
            max_entity_tokens=1000,
            max_relation_tokens=1000,
            max_total_tokens=3000
        )
        context = await self.rag.aquery(query=query, param=param)
        clean_context = extract_descriptions_lightrag(str(context)) # type: ignore
        cost = calculate_total_cost(Config.TOKEN_TRACKER.get_usage()) # type: ignore
        Config.TOKEN_TRACKER.reset() # type: ignore
        return clean_context, cost


class HippoRAGAdapter(BaseRAGAdapter):
    def __init__(self, hipporag_instance):
        self.hipporag = hipporag_instance

    async def index(self, documents: list[str]):
        await asyncio.to_thread(self.hipporag.index, docs=documents)
        return 0.0

    @retry(wait=wait_exponential(1, max=10), stop=stop_after_attempt(5))
    async def retrieve(self, query: str) -> tuple[list[str], float]:
        if not self.hipporag.fact_embedding_store.embeddings:
            return [], 0.0

        # spend_before = get_litellm_usage()
        results = await asyncio.to_thread(self.hipporag.retrieve, queries=[query])
        # spend_after = get_litellm_usage()
        context = results[0].docs[:5]
        # cost = spend_after - spend_before

        return context, 0.0

class SpannerGraphRAGAdapter(BaseRAGAdapter):
    def __init__(self, graph_store, embedding_service, llm_transformer, graph_name, gemini_client):
        self.graph_store = graph_store
        self.embedding_service = embedding_service
        self.llm_transformer = llm_transformer
        self.graph_name = graph_name
        self.gemini_client = gemini_client

    @retry(wait=wait_exponential(min=4, max=30), stop=stop_after_attempt(15))
    async def _safe_extract_graph(self, doc: Document, config=None):
        # config carries the UsageTrackingCallback so extraction LLM cost is captured.
        res = await self.llm_transformer.aconvert_to_graph_documents([doc], config=config)
        return res[0]

    async def index(self, documents: list[str]) -> float:
        print("Cleaning up graph store before indexing...")
        await asyncio.to_thread(self.graph_store.cleanup)
        
        document_list = load_documents(documents)
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
                    return doc.page_content, graph_doc
                except Exception as e:
                    print(f"Failed to extract graph for document {idx+1}: {e}")
                    return doc.page_content, None

        tasks = [extract_with_semaphore(i, doc) for i, doc in enumerate(document_list)]
        results = await asyncio.gather(*tasks)
        
        graph_documents_with_chunks = [(text, graph_doc) for text, graph_doc in results if graph_doc is not None]
            
        # Ensure consistent types for each node ID across all documents
        global_node_types = {}
        for chunk_text, doc in graph_documents_with_chunks:
            for node in doc.nodes:
                if not getattr(node, "type", None) or str(node.type).strip().lower() in ["null", "none", ""]:
                    node.type = "Unknown"
                if node.id not in global_node_types:
                    global_node_types[node.id] = node.type
            for rel in doc.relationships:
                for target_node in [rel.source, rel.target]:
                    if not getattr(target_node, "type", None) or str(target_node.type).strip().lower() in ["null", "none", ""]:
                        target_node.type = "Unknown"
                    if target_node.id not in global_node_types:
                        global_node_types[target_node.id] = target_node.type

        valid_graph_documents = []
        for chunk_text, doc in graph_documents_with_chunks:
            existing_node_ids = set()
            new_nodes = []
            
            for node in doc.nodes:
                node.type = global_node_types[node.id]
                if node.id not in existing_node_ids:
                    new_nodes.append(node)
                    existing_node_ids.add(node.id)
                    
            for rel in doc.relationships:
                if not getattr(rel, "type", None) or str(rel.type).strip().lower() in ["null", "none", ""]:
                    rel.type = "RELATED_TO"
                    
                rel.source.type = global_node_types[rel.source.id]
                rel.target.type = global_node_types[rel.target.id]
                
                if rel.source.id not in existing_node_ids:
                    new_nodes.append(rel.source)
                    existing_node_ids.add(rel.source.id)
                if rel.target.id not in existing_node_ids:
                    new_nodes.append(rel.target)
                    existing_node_ids.add(rel.target.id)
            
            doc.nodes = new_nodes
            valid_graph_documents.append(doc)
        
        texts_to_embed = []
        node_references = []
        
        for chunk_text, graph_document in graph_documents_with_chunks:
            chunk_id = f"Chunk_{hashlib.md5(chunk_text.encode('utf-8')).hexdigest()}"
            chunk_node = Node(id=chunk_id, type="Chunk", properties={"text": chunk_text})
            
            texts_to_embed.append(chunk_text[:1000])
            node_references.append(chunk_node)
                
            original_nodes = list(graph_document.nodes)
            for node in original_nodes:
                texts_to_embed.append(node.id)
                node_references.append(node)
                
                rel = Relationship(source=node, target=chunk_node, type="MENTIONED_IN")
                graph_document.relationships.append(rel)
                    
            graph_document.nodes.append(chunk_node)

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
            # Consolidate all nodes and relationships into a single GraphDocument
            # to prevent duplicate DDL schema generation errors in Spanner.
            global_nodes = {}
            for doc in valid_graph_documents:
                for node in doc.nodes:
                    if node.id not in global_nodes:
                        global_nodes[node.id] = node
                    else:
                        if node.properties:
                            if not global_nodes[node.id].properties:
                                global_nodes[node.id].properties = {}
                            global_nodes[node.id].properties.update(node.properties)

            global_relationships = []
            seen_relationships = set()
            for doc in valid_graph_documents:
                for rel in doc.relationships:
                    rel.source = global_nodes[rel.source.id]
                    rel.target = global_nodes[rel.target.id]
                    
                    rel_key = (rel.source.id, rel.target.id, rel.type)
                    if rel_key not in seen_relationships:
                        seen_relationships.add(rel_key)
                        global_relationships.append(rel)

            merged_doc = GraphDocument(
                nodes=list(global_nodes.values()),
                relationships=global_relationships,
                source=valid_graph_documents[0].source
            )

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
            LIMIT 5
            RETURN SAFE_TO_JSON(node) as node_json
        """
        
        responses = await asyncio.to_thread(self.graph_store.query, gql_query)
        chunk_texts = set()
        
        async def fetch_connected_chunks(node_id, label_str, edge_label):
            chunk_query = f"""
                GRAPH {self.graph_name}
                MATCH (node{label_str})-[e:{edge_label}]-(chunk:Chunk)
                WHERE node.id = '{node_id}'
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

        tasks = []
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
                
                tasks.append(fetch_connected_chunks(node_id, label_str, edge_label))
                    
            except Exception as e:
                print(f"Failed to process top node: {e}")
                continue
        
        if tasks:
            nested_texts = await asyncio.gather(*tasks)
            for texts in nested_texts:
                for text in texts:
                    chunk_texts.add(text)

        return list(chunk_texts)[:5], cost
    
class AgenticRAGAdapter(BaseRAGAdapter):
    def __init__(self, qdrant_client, embedding_service, gemini_client):
        self.qdrant_client = qdrant_client
        self.gemini_client = gemini_client
        self._query_costs = []
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
        workflow.add_node(self.generate_query_or_respond)
        workflow.add_node("_retrieve", ToolNode([self.retrieve_context]))
        workflow.add_node(self.rewrite_question)
        workflow.add_node(self.generate_answer)

        workflow.add_edge(START, "generate_query_or_respond")
        workflow.add_conditional_edges(
            "generate_query_or_respond",
            self.route_on_tool_calls,
            {
                "tools": "_retrieve",
                END:END
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

        result = await self.graph.ainvoke({"messages": [SystemMessage(content=Config.RAG_SYSTEM_PROMPT), HumanMessage(content=query)]})

        # Split each tool message back into its chunks (joined with "\n\n" in
        # _sync_retrieve_context) and dedup across all retrieval rounds so
        # partially-overlapping rounds don't pile up duplicates.
        raw_chunks = []
        for message in result["messages"]:
            if isinstance(message, ToolMessage):
                raw_chunks.extend(str(message.content).split("\n\n"))
        context_blocks = dedup_preserve_order(raw_chunks)

        final_answer = ""

        for message in reversed(result["messages"]):
            if isinstance(message, AIMessage) and message.content:
                final_answer = str(message.content)
                break

        self.responses_dict[query] = final_answer
        return context_blocks, sum(self._query_costs)


    def get_response(self, query: str) -> str:
        return self.responses_dict.get(query, "")
    

    def _sync_retrieve_context(self, query: str) -> str:
        # Embed the query through the gateway (capturing cost), then search by the
        # precomputed vector. Over-fetch, then trim to 5 distinct chunks (shared
        # "vector_storage" collection, so residual duplicate points are possible).
        vector, cost = embed_query_with_cost(self.gemini_client, Config.EMBEDDING_MODEL, query) # type: ignore
        self._query_costs.append(cost)
        search_result = self.vector_store.similarity_search_by_vector(vector, k=20)
        retrieved_context = dedup_preserve_order([doc.page_content for doc in search_result])[:5]
        return "\n\n".join(retrieved_context)

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
        context_blocks = []
        for msg in reversed(state["messages"]):
            if not hasattr(msg, "tool_call_id"):
                break
            context_blocks.append(str(msg.content))
        
        context = "\n\n".join(context_blocks[::-1]) 
     
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
        context_blocks = []
        for msg in reversed(state["messages"]):
            if not hasattr(msg, "tool_call_id"):
                break
            context_blocks.append(str(msg.content))

        context = "\n\n".join(context_blocks[::-1])
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
        context_blocks = []
        for msg in reversed(state["messages"]):
            if not hasattr(msg, "tool_call_id"):
                break
            context_blocks.append(str(msg.content))
        
        context = "\n\n".join(context_blocks[::-1])
        prompt = Config.GENERATE_PROMPT.format(question=question, context=context)
        response = await self.response_model.ainvoke([{"role": "user", "content": prompt}])
        self._query_costs.append(message_cost(response))

        return {"messages": [response]}



    def route_on_tool_calls(self, state: MessagesState):
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "tools"
    
        return END
    
