"""Corpus loading and text hygiene: pptx parsing, Document wrapping, dedup."""
import os
import tiktoken
from langchain_core.documents import Document
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker

_ENCODER = tiktoken.get_encoding("cl100k_base")


def load_documents(chunks: list[str]) -> list[Document]:
    return [Document(chunk) for chunk in chunks]


def normalize_text(text: str) -> str:
    # collapse whitespace so whitespace-only variants dedup together
    return " ".join(text.split())


def truncate_to_token_budget(items: list[str], max_tokens: int) -> list[str]:
    # Keep whole items until the budget is hit; hard-trim only if item 0 alone overflows.
    out, used = [], 0
    for item in items:
        toks = _ENCODER.encode(item)
        if used + len(toks) <= max_tokens:
            out.append(item)
            used += len(toks)
        elif not out:
            out.append(_ENCODER.decode(toks[:max_tokens]))
            break
        else:
            break
    return out


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
        name = os.path.basename(path)
        for chunk in chunks:
            text = f"[source: {name}] {chunk.text}"
            if text not in context:
                context.append(text)

    return context


def fuse_strings(contents: list[str], min_tokens: int) -> list[str]:
    sorted_contents = sorted(contents, key=lambda c: len(_ENCODER.encode(c)))

    fused_list: list[str] = []
    current = ""

    for content in sorted_contents:
        if len(_ENCODER.encode(current)) > min_tokens:
            fused_list.append(current)
            current = content
        else:
            current = f"{current} {content}" if current else content

    if current:
        fused_list.append(current)

    return fused_list
