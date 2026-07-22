"""Indexing checkpoints so a crash mid-run resumes without re-indexing.

Indexing is the expensive phase (LLM graph/OpenIE extraction). run_pipeline
persists two artifacts per strategy under results/{label}/:

  - index_manifest.json : {adapter_name: indexing_cost}, written incrementally as
    each adapter finishes indexing. On resume the already-indexed adapters are
    skipped and their cost reused; the underlying stores (Qdrant / Neo4j /
    Spanner / HippoRAG save_dir) still hold this strategy's data because
    completed strategies are reloaded (below) and later strategies have not run.

  - run_result.pkl : the full run_pipeline return dict, written LAST as an
    atomic-ish completion marker. On resume a fully-completed strategy is
    reloaded verbatim (no adapters built, no LLM calls), so its shared stores are
    never re-touched -- which is what keeps the manifest skip above sound.

Both artifacts embed a content fingerprint; a mismatch (corpus changed without a
fresh Config.setup_directories, which wipes results/) invalidates the checkpoint
and forces a clean rebuild.
"""
import os
import json
import pickle
import hashlib

_MANIFEST = "index_manifest.json"
_RESULT = "run_result.pkl"


def content_fingerprint(content_list: list[str]) -> str:
    # Order-sensitive hash of the corpus; length-prefixed + NUL-delimited so
    # concatenation boundaries can't collide.
    h = hashlib.sha256()
    h.update(f"{len(content_list)}\x00".encode("utf-8"))
    for c in content_list:
        h.update(c.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def load_index_manifest(results_dir: str, content_hash: str) -> dict[str, float]:
    path = os.path.join(results_dir, _MANIFEST)
    try:
        with open(path) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    if data.get("content_hash") != content_hash:
        return {}  # stale corpus; ignore
    return {k: float(v) for k, v in data.get("adapters", {}).items()}


def save_index_entry(results_dir: str, content_hash: str, name: str, cost: float) -> None:
    os.makedirs(results_dir, exist_ok=True)
    adapters = load_index_manifest(results_dir, content_hash)
    adapters[name] = cost
    tmp = os.path.join(results_dir, _MANIFEST + ".tmp")
    with open(tmp, "w") as f:
        json.dump({"content_hash": content_hash, "adapters": adapters}, f)
    os.replace(tmp, os.path.join(results_dir, _MANIFEST))


def load_completed_run(results_dir: str, content_hash: str) -> dict | None:
    path = os.path.join(results_dir, _RESULT)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        result = pickle.load(f)
    if result.get("content_hash") != content_hash:
        return None  # stale corpus
    return result


def save_completed_run(results_dir: str, result: dict) -> None:
    os.makedirs(results_dir, exist_ok=True)
    tmp = os.path.join(results_dir, _RESULT + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(result, f)
    os.replace(tmp, os.path.join(results_dir, _RESULT))
    # Manifest is now redundant; drop it so bookkeeping stays clean.
    manifest = os.path.join(results_dir, _MANIFEST)
    if os.path.exists(manifest):
        os.remove(manifest)
