"""Embed a dataset's chunks and queries with the dense text model, one worker per GPU. Resumable.

Usage:
    python scripts/embed_text.py --dataset v2_esg_reports [--corpus NAME] [--workers 2]

Reads <DATA_ROOT>/derived/chunks/<corpus>/chunks.jsonl and the dataset's queries; the corpus defaults to the
dataset. Writes <DATA_ROOT>/derived/embeddings/<corpus>/<model>/{chunks,queries}_w<i>.npz and
<RESULTS_ROOT>/embeddings/<corpus>/: config.yaml, run_info.json, summary.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from doc_qa.core.config import get_settings, load_yaml
from doc_qa.core.parallel import launch_workers, read_jsonl
from doc_qa.core.run_info import gpu_names, write_run_info
from doc_qa.evaluation.vidore import load_benchmark
from doc_qa.retrieval.dense import embed_texts, load_embedder, load_vectors, save_vectors


def load_inputs(dataset: str, corpus: str, cfg: dict[str, Any]) -> dict[str, tuple[list[str], list[str], str | None]]:
    """Collect the texts to embed.

    Args:
        dataset (str): Dataset key (queries).
        corpus (str): Chunk folder name.
        cfg (dict[str, Any]): Text config.

    Returns:
        dict[str, tuple[list[str], list[str], str | None]]: {kind: (ids, texts, prompt name)}.
    """
    settings = get_settings()
    chunks = read_jsonl([settings.data_root / "derived" / "chunks" / corpus / "chunks.jsonl"])
    if not chunks:
        raise SystemExit(f"No chunks for {corpus} (run build_chunks.py first).")
    spec = load_yaml(Path(cfg["datasets_config"]))["datasets"][dataset]
    queries = sorted(load_benchmark(dataset, spec, settings.data_root / "raw").queries.values(), key=lambda q: q.query_id)
    return {
        "chunks": ([c["chunk_id"] for c in chunks], [c["text"] for c in chunks], None),
        "queries": ([q.query_id for q in queries], [q.text for q in queries], cfg["query_prompt_name"]),
    }


def run_worker(args: argparse.Namespace, cfg: dict[str, Any], out_dir: Path) -> None:
    """Embed this worker's share of chunks and queries; skip kinds already done.

    Args:
        args (argparse.Namespace): CLI arguments (with worker index and count).
        cfg (dict[str, Any]): Text config.
        out_dir (Path): Embedding folder.
    """
    tag, embedder = f"[worker {args.worker_index}]", None
    for kind, (ids, texts, prompt) in load_inputs(args.dataset, args.corpus, cfg).items():
        path = out_dir / f"{kind}_w{args.worker_index}.npz"
        if path.exists():
            print(f"{tag} {kind}: already done", flush=True)
            continue
        share = slice(args.worker_index, None, args.num_workers)
        if embedder is None:
            embedder = load_embedder(cfg["embedding_model"], cfg["embedding_revision"], cfg["max_tokens"])
        start = time.perf_counter()
        vectors = embed_texts(embedder, texts[share], cfg["embed_batch_size"], prompt)
        save_vectors(path, ids[share], vectors)
        print(f"{tag} {kind}: {len(vectors)} in {time.perf_counter() - start:.0f} s", flush=True)


def main() -> None:
    """Launch one embedding worker per GPU, then check that every chunk and query has a vector."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/text.yaml"))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--corpus", help="Chunk folder name (default: the dataset key).")
    parser.add_argument("--workers", type=int, help="Default: number of GPUs (1 on CPU).")
    parser.add_argument("--worker-index", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--num-workers", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.corpus = args.corpus or args.dataset

    settings = get_settings()
    cfg = load_yaml(args.config)
    out_dir = settings.data_root / "derived" / "embeddings" / args.corpus / cfg["embedding_model"].split("/")[-1]
    if args.worker_index is not None:
        run_worker(args, cfg, out_dir)
        return

    from huggingface_hub import snapshot_download

    snapshot_download(cfg["embedding_model"], revision=cfg["embedding_revision"])  # once, before workers start
    results_dir = settings.results_root / "embeddings" / args.corpus
    write_run_info(results_dir, extra={
        "dataset": args.dataset, "corpus": args.corpus,
        "model": {"repo": cfg["embedding_model"], "revision": cfg["embedding_revision"]},
    })
    (results_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    start = time.perf_counter()
    workers = args.workers or max(1, len(gpu_names()))
    cmd = [sys.executable, str(Path(__file__).resolve()), "--config", str(args.config),
           "--dataset", args.dataset, "--corpus", args.corpus]
    launch_workers(cmd, workers)

    summary: dict[str, Any] = {"workers": workers, "seconds": round(time.perf_counter() - start)}
    for kind, (ids, _, _) in load_inputs(args.dataset, args.corpus, cfg).items():
        vectors = load_vectors(sorted(out_dir.glob(f"{kind}_w*.npz")))
        missing = len(set(ids) - set(vectors))
        if missing:
            raise SystemExit(f"{missing} {kind} have no vector; rerun to finish.")
        summary[kind] = len(ids)
        summary["dim"] = len(next(iter(vectors.values())))
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
