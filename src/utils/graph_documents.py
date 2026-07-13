"""Validation and merging of extracted GraphDocuments for graph indexing."""
import hashlib
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship


def _is_missing_type(value) -> bool:
    # LLM extraction sometimes emits null-ish node/relationship types
    return not value or str(value).strip().lower() in ["null", "none", ""]


def build_global_node_types(graph_documents_with_chunks: list[tuple[str, GraphDocument]]) -> dict[str | int, str]:
    """Build a node.id -> type registry across all extracted documents.

    Normalizes null-ish types to "Unknown" (mutating nodes and relationship
    endpoints in place); the first type seen for an id wins, so the same
    entity keeps a consistent type across documents.
    """
    global_node_types: dict[str | int, str] = {}
    for _, doc in graph_documents_with_chunks:
        for node in doc.nodes:
            if _is_missing_type(getattr(node, "type", None)):
                node.type = "Unknown"
            if node.id not in global_node_types:
                global_node_types[node.id] = node.type
        for rel in doc.relationships:
            for endpoint in [rel.source, rel.target]:
                if _is_missing_type(getattr(endpoint, "type", None)):
                    endpoint.type = "Unknown"
                if endpoint.id not in global_node_types:
                    global_node_types[endpoint.id] = endpoint.type
    return global_node_types


def sanitize_graph_documents(graph_documents_with_chunks: list[tuple[str, GraphDocument]],
                             global_node_types: dict[str | int, str]) -> list[GraphDocument]:
    """Validate each document's nodes and relationships in place.

    Applies the global type registry, dedups nodes by id, defaults missing
    relationship types to "RELATED_TO", and appends relationship endpoints
    missing from doc.nodes. Returns the same (mutated) GraphDocument objects.
    """
    valid_graph_documents = []
    for _, doc in graph_documents_with_chunks:
        existing_node_ids = set()
        new_nodes = []

        for node in doc.nodes:
            node.type = global_node_types[node.id]
            if node.id not in existing_node_ids:
                new_nodes.append(node)
                existing_node_ids.add(node.id)

        for rel in doc.relationships:
            if _is_missing_type(getattr(rel, "type", None)):
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
    return valid_graph_documents


def attach_chunk_nodes(graph_documents_with_chunks: list[tuple[str, GraphDocument]]) -> tuple[list[str], list[Node]]:
    """Attach a Chunk node per document and link every entity to it.

    Adds a content-hash Chunk node ("Chunk_<md5>") and a MENTIONED_IN
    relationship from each entity node (mutating the documents in place).
    Returns (texts_to_embed, node_references) aligned pairwise so callers can
    embed the texts and assign each vector onto the referenced node — chunk
    text is truncated to 1000 chars for embedding only.
    """
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
    return texts_to_embed, node_references


def merge_graph_documents(valid_graph_documents: list[GraphDocument]) -> GraphDocument:
    """Consolidate all documents into a single GraphDocument.

    Merges nodes by id (union-ing properties into the first occurrence),
    rewires relationship endpoints to the merged nodes, and dedups
    relationships on (source.id, target.id, type). A single document prevents
    duplicate DDL schema generation errors in Spanner.
    """
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

    return GraphDocument(
        nodes=list(global_nodes.values()),
        relationships=global_relationships,
        source=valid_graph_documents[0].source
    )
