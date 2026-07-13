"""Corpus loading and text hygiene: pptx parsing, Document wrapping, dedup."""
import os
from langchain_core.documents import Document
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker


def load_documents(chunks: list[str]) -> list[Document]:
    return [Document(chunk) for chunk in chunks]


def normalize_text(text: str) -> str:
    # collapse whitespace so whitespace-only variants dedup together
    return " ".join(text.split())


def dedup_preserve_order(items: list[str]) -> list[str]:
    seen, out = set(), []
    for item in items:
        key = normalize_text(item)
        if key and key not in seen:
            seen.add(key)
            out.append(item)  # keep original text, dedup on normalized key
    return out


def process_pptx_file(paths: str | list[str]) -> list[str]:
    if isinstance(paths, str):
        paths = [paths]

    converter = DocumentConverter()
    chunker = HierarchicalChunker()
    context = []

    for path in paths:
        if not os.path.isfile(path):
            raise OSError(f"File not found: {path}")
        result = converter.convert(path)
        chunks = chunker.chunk(result.document)
        for chunk in chunks:
            if chunk.text not in context:
                context.append(chunk.text)

    return context
