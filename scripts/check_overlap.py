"""Check dev datasets (ViDoRe V1/V2) for overlap with the test set (ViDoRe V3).

Usage:
    python scripts/check_overlap.py --config configs/overlap.yaml

Writes to <RESULTS_ROOT>/<name>/: config.yaml, run_info.json, report.md, report.json,
proposed_exclusions.yaml, artifacts/pairs/ (confirmed), artifacts/pairs_rejected/ (closest rejected).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from doc_qa.core.config import get_settings, load_yaml
from doc_qa.core.run_info import write_run_info
from doc_qa.core.schemas import Benchmark
from doc_qa.evaluation.overlap import (
    PageHash,
    find_exact_matches,
    find_phash_matches,
    find_title_matches,
    hash_pages,
    load_hashes,
    page_views,
    save_hashes,
    side_by_side,
    verify_pair,
)
from doc_qa.evaluation.vidore import apply_exclusions, iter_page_images, load_benchmarks


def page_hashes(bench: Benchmark, revision: str, cache_dir: Path, workers: int | None) -> list[PageHash]:
    """Hash a dataset's pages, cached per dataset revision.

    Args:
        bench (Benchmark): Dataset.
        revision (str): Dataset revision.
        cache_dir (Path): Cache folder.
        workers (int | None): Hashing processes; None = all cores.

    Returns:
        list[PageHash]: One per page.
    """
    path = cache_dir / f"{bench.name}@{revision[:12]}.csv"
    if path.exists():
        cached = load_hashes(path)
        if {h.page_id for h in cached} >= set(bench.pages):
            return [h for h in cached if h.page_id in bench.pages]
    print(f"Hashing {len(bench.pages)} pages of {bench.name} ...", flush=True)
    hashes = hash_pages(iter_page_images(bench), workers=workers)
    save_hashes(hashes, path)
    return hashes


def summarize(bench: Benchmark) -> dict[str, Any]:
    """Count a dataset's documents, pages, queries and qrels.

    Args:
        bench (Benchmark): Dataset.

    Returns:
        dict[str, Any]: Counts and warnings.
    """
    languages = Counter(q.language for q in bench.queries.values())
    return {
        "dataset": bench.name,
        "role": bench.role,
        "documents": len(bench.documents),
        "pages": len(bench.pages),
        "queries_en": languages.get("en", 0),
        "queries_fr": languages.get("fr", 0),
        "queries_with_answers": sum(1 for q in bench.queries.values() if q.answers),
        "qrels": len(bench.qrels),
        "warnings": bench.warnings,
    }


def collect_page_pairs(
    exact: list[tuple[str, str]], near: list[tuple[str, str, int]]
) -> list[dict[str, Any]]:
    """Merge exact and pHash matches into one pair list, exact first, then by distance.

    Args:
        exact (list[tuple[str, str]]): (dev page, test page).
        near (list[tuple[str, str, int]]): (dev page, test page, distance).

    Returns:
        list[dict[str, Any]]: Pairs with dev_page, test_page, exact, phash_distance.
    """
    exact_set = set(exact)
    distance = {(d, t): dist for d, t, dist in near}
    pairs = [
        {"dev_page": d, "test_page": t, "exact": (d, t) in exact_set, "phash_distance": distance.get((d, t))}
        for d, t in exact_set | set(distance)
    ]
    return sorted(pairs, key=lambda p: (not p["exact"], p["phash_distance"] or 0, p["dev_page"], p["test_page"]))


def _pages_by_dataset(pairs: list[dict[str, Any]]) -> dict[str, set[str]]:
    """Group the pairs' page ids by dataset key.

    Args:
        pairs (list[dict[str, Any]]): Page pairs.

    Returns:
        dict[str, set[str]]: Page ids per dataset.
    """
    needed: dict[str, set[str]] = defaultdict(set)
    for p in pairs:
        for page_id in (p["dev_page"], p["test_page"]):
            needed[page_id.split("/", 1)[0]].add(page_id)
    return needed


def verify_pairs(pairs: list[dict[str, Any]], benches: list[Benchmark], cfg: dict[str, Any]) -> None:
    """Add status, ink_share and correlation to each pair, in place.

    Args:
        pairs (list[dict[str, Any]]): Candidate pairs.
        benches (list[Benchmark]): All datasets.
        cfg (dict[str, Any]): Config (verify_size, min_ink_share, min_pixel_correlation, workers).
    """
    by_name = {b.name: b for b in benches}
    views = {}
    for name, page_ids in _pages_by_dataset(pairs).items():
        print(f"Reading {len(page_ids)} candidate pages of {name} ...", flush=True)
        views.update(page_views(iter_page_images(by_name[name], page_ids), cfg["verify_size"], cfg["workers"]))
    for p in pairs:
        dev, test = views[p["dev_page"]], views[p["test_page"]]
        status, correlation = verify_pair(dev, test, cfg["min_ink_share"], cfg["min_pixel_correlation"])
        p["status"] = status
        p["ink_share"] = round(min(dev.ink_share, test.ink_share), 4)
        p["correlation"] = None if correlation is None else round(correlation, 3)


def flag_documents(
    pairs: list[dict[str, Any]], titles: list[tuple[str, str, float, str]], page_doc: dict[str, str]
) -> dict[str, dict[str, Any]]:
    """Group matched pages and titles by dev document.

    Args:
        pairs (list[dict[str, Any]]): Confirmed page pairs.
        titles (list[tuple[str, str, float, str]]): Title/URL matches.
        page_doc (dict[str, str]): Page id -> document id.

    Returns:
        dict[str, dict[str, Any]]: Evidence per dev document id.
    """
    evidence: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"test_documents": set(), "exact_pages": set(), "near_pages": set(), "title_matches": []}
    )
    for p in pairs:
        e = evidence[page_doc[p["dev_page"]]]
        e["test_documents"].add(page_doc[p["test_page"]])
        e["exact_pages" if p["exact"] else "near_pages"].add(p["dev_page"])
    for dev_doc, test_doc, ratio, kind in titles:
        e = evidence[dev_doc]
        e["test_documents"].add(test_doc)
        e["title_matches"].append({"test_document": test_doc, "similarity": ratio, "kind": kind})
    return {
        doc: {
            "test_documents": sorted(e["test_documents"]),
            "exact_pages": len(e["exact_pages"]),
            "near_pages": len(e["near_pages"] - e["exact_pages"]),
            "title_matches": e["title_matches"],
        }
        for doc, e in sorted(evidence.items())
    }


def removal_impact(dev: list[Benchmark], excluded: list[str], large_share: float) -> list[dict[str, Any]]:
    """Compute what each dev dataset loses if `excluded` documents are removed.

    Args:
        dev (list[Benchmark]): Dev datasets.
        excluded (list[str]): Document ids.
        large_share (float): Lost-query share above which the overlap is large.

    Returns:
        list[dict[str, Any]]: Before/after counts per dataset.
    """
    rows = []
    for bench in dev:
        after = apply_exclusions(bench, excluded)
        lost = 1 - len(after.queries) / len(bench.queries) if bench.queries else 0.0
        rows.append(
            {
                "dataset": bench.name,
                "documents": [len(bench.documents), len(after.documents)],
                "pages": [len(bench.pages), len(after.pages)],
                "queries": [len(bench.queries), len(after.queries)],
                "query_share_lost": round(lost, 3),
                "large": lost > large_share,
            }
        )
    return rows


def _safe_name(page_id: str, max_len: int = 60) -> str:
    """Make a page id file-name safe.

    Args:
        page_id (str): Page id.
        max_len (int): Max length (keeps the end).

    Returns:
        str: Safe name.
    """
    return re.sub(r"[^\w.-]+", "_", page_id)[-max_len:]


def save_pair_images(
    pairs: list[dict[str, Any]], benches: list[Benchmark], out_dir: Path, subdir: str, limit: int
) -> int:
    """Save up to `limit` pairs side by side to artifacts/<subdir>/ (cleared first).

    Args:
        pairs (list[dict[str, Any]]): Pairs, in save order.
        benches (list[Benchmark]): All datasets.
        out_dir (Path): Experiment folder.
        subdir (str): Folder under artifacts/.
        limit (int): Max images.

    Returns:
        int: Images saved.
    """
    pair_dir = out_dir / "artifacts" / subdir
    if pair_dir.exists():
        shutil.rmtree(pair_dir)
    chosen = pairs[:limit]
    if not chosen:
        return 0
    by_name = {b.name: b for b in benches}
    images: dict[str, bytes] = {}
    for name, page_ids in _pages_by_dataset(chosen).items():
        images.update(iter_page_images(by_name[name], page_ids))

    pair_dir.mkdir(parents=True)
    for n, p in enumerate(chosen):
        tag = f"r{p['correlation']:.2f}_" + ("exact" if p["exact"] else f"d{p['phash_distance']:02d}")
        file_name = f"{n:03d}_{tag}_{_safe_name(p['dev_page'])}__{_safe_name(p['test_page'])}.png"
        side_by_side(images[p["dev_page"]], images[p["test_page"]]).save(pair_dir / file_name)
        p["image"] = f"artifacts/{subdir}/{file_name}"
    return len(chosen)


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    """Format a Markdown table.

    Args:
        headers (list[str]): Column names.
        rows (list[list[Any]]): Rows.

    Returns:
        str: Table.
    """
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def verdict(flagged: dict[str, Any], impact: list[dict[str, Any]]) -> str:
    """One-line verdict.

    Args:
        flagged (dict[str, Any]): Flagged dev documents.
        impact (list[dict[str, Any]]): Removal impact per dev dataset.

    Returns:
        str: Verdict.
    """
    large = [r["dataset"] for r in impact if r["large"]]
    if not flagged:
        return "No overlap found: the dev set can be used as is."
    if large:
        return f"Large overlap in {', '.join(large)}: discuss before using V1/V2 as dev."
    return "Small overlap: review the pairs, then copy the confirmed ids into configs/dev_exclusions.yaml."


def write_report(out_dir: Path, report: dict[str, Any]) -> None:
    """Write report.json, report.md and proposed_exclusions.yaml.

    Args:
        out_dir (Path): Experiment folder.
        report (dict[str, Any]): Results.
    """
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    proposed = {"ids": list(report["flagged_documents"]), "evidence": report["flagged_documents"]}
    with (out_dir / "proposed_exclusions.yaml").open("w", encoding="utf-8") as f:
        f.write("# Review artifacts/pairs/, then copy confirmed ids into configs/dev_exclusions.yaml.\n")
        yaml.safe_dump(proposed, f, sort_keys=False, allow_unicode=True)

    m, cfg = report["matches"], report["config"]
    datasets = [
        [s["dataset"], s["role"], s["documents"], s["pages"], s["queries_en"], s["queries_fr"],
         s["queries_with_answers"], s["qrels"], "; ".join(s["warnings"]) or "-"]
        for s in report["datasets"]
    ]
    flagged = [
        [doc, ", ".join(e["test_documents"]), e["exact_pages"], e["near_pages"], len(e["title_matches"])]
        for doc, e in report["flagged_documents"].items()
    ]
    impact = [
        [r["dataset"], "{} → {}".format(*r["documents"]), "{} → {}".format(*r["pages"]),
         "{} → {}".format(*r["queries"]), f"{r['query_share_lost']:.1%}", "yes" if r["large"] else "no"]
        for r in report["impact_if_all_removed"]
    ]
    current = [
        [s["dataset"], s["documents"], s["pages"], s["queries_en"] + s["queries_fr"]]
        for s in report["dev_after_current_exclusions"]["datasets"]
    ]
    unknown = report["dev_after_current_exclusions"]["unknown_ids"]

    md = [
        "# Overlap check: dev (ViDoRe V1/V2) vs test (ViDoRe V3)",
        "",
        f"**Verdict:** {report['verdict']}",
        "",
        "## Datasets (English and French queries)",
        _md_table(["dataset", "role", "docs", "pages", "queries en", "queries fr", "with answers", "qrels", "warnings"], datasets),
        "",
        "## Matches",
        f"Candidate page pairs: {m['candidate_pairs']} (identical pixels: {m['exact_pages']}, "
        f"pHash distance ≤ {cfg['phash_max_distance']}: {m['near_pages']})",
        f"- Skipped, nearly blank page (ink < {cfg['min_ink_share']:.0%} of pixels): {m['blank']}",
        f"- Rejected, different pages (pixel correlation < {cfg['min_pixel_correlation']}): {m['rejected']}",
        f"- **Confirmed, same page: {m['confirmed']}**",
        "",
        f"Title or URL matches: {m['titles']}",
        "",
        f"Images saved: {m['images_saved']} confirmed pairs (artifacts/pairs/), "
        f"{m['rejected_images_saved']} rejected pairs closest to the threshold (artifacts/pairs_rejected/)",
        "",
        "## Flagged dev documents (confirmed pages or matching titles)",
        _md_table(["dev document", "test documents", "exact pages", "near pages", "title matches"], flagged) if flagged else "None.",
        "",
        "## Impact if every flagged document is removed",
        _md_table(["dataset", "documents", "pages", "queries", "queries lost", "large"], impact),
        "",
        f"## Dev after the exclusions file (documents excluded: {report['dev_after_current_exclusions']['excluded']})",
        _md_table(["dataset", "documents", "pages", "queries"], current),
    ]
    if unknown:
        md += ["", f"**Unknown ids in configs/dev_exclusions.yaml:** {', '.join(unknown)}"]
    (out_dir / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> None:
    """Run the overlap check."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/overlap.yaml"))
    args = parser.parse_args()

    settings = get_settings()
    cfg = load_yaml(args.config)
    specs = load_yaml(Path(cfg["datasets_config"]))["datasets"]
    out_dir = settings.results_root / cfg["name"]
    write_run_info(out_dir, extra={"datasets": {k: {"repo": s["repo"], "revision": s["revision"]} for k, s in specs.items()}})
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    # Load without exclusions: the check needs the full dev set.
    benches = load_benchmarks(specs, settings.data_root / "raw", cfg["languages"])
    dev = [b for b in benches if b.role == "dev"]
    test = [b for b in benches if b.role == "test"]
    summaries = [summarize(b) for b in benches]
    for s in summaries:
        print(f"{s['dataset']:<26} {s['role']:<4} pages={s['pages']:>5} queries={s['queries_en'] + s['queries_fr']:>5}")

    cache_dir = settings.data_root / "derived" / "page_hashes"
    hashes = {b.name: page_hashes(b, specs[b.name]["revision"], cache_dir, cfg["workers"]) for b in benches}
    dev_hashes = [h for b in dev for h in hashes[b.name]]
    test_hashes = [h for b in test for h in hashes[b.name]]

    # Candidate pairs from hashes, then pixel verification.
    exact = find_exact_matches(dev_hashes, test_hashes)
    near = find_phash_matches(dev_hashes, test_hashes, cfg["phash_max_distance"], cfg["max_aspect_ratio_diff"])
    pairs = collect_page_pairs(exact, near)
    verify_pairs(pairs, benches, cfg)
    statuses = Counter(p["status"] for p in pairs)
    confirmed = [p for p in pairs if p["status"] == "confirmed"]
    rejected = sorted((p for p in pairs if p["status"] == "rejected"), key=lambda p: -p["correlation"])

    titles = find_title_matches(
        [d for b in dev for d in b.documents.values()],
        [d for b in test for d in b.documents.values()],
        cfg["title_min_ratio"],
    )
    page_doc = {pid: p.doc_id for b in benches for pid, p in b.pages.items()}
    flagged = flag_documents(confirmed, titles, page_doc)
    impact = removal_impact(dev, list(flagged), cfg["large_overlap_query_share"])
    images_saved = save_pair_images(confirmed, benches, out_dir, "pairs", cfg["max_pair_images"])
    rejected_saved = save_pair_images(rejected, benches, out_dir, "pairs_rejected", cfg["max_rejected_images"])

    # Dev as served by the loader, with current exclusions.
    current = load_yaml(Path(cfg["exclusions_config"])).get("documents") or []
    dev_doc_ids = {d for b in dev for d in b.documents}
    after = [summarize(apply_exclusions(b, current)) for b in dev]

    report = {
        "verdict": verdict(flagged, impact),
        "config": cfg,
        "datasets": summaries,
        "matches": {
            "exact_pages": len(exact),
            "near_pages": len(near),
            "candidate_pairs": len(pairs),
            "blank": statuses["blank"],
            "rejected": statuses["rejected"],
            "confirmed": statuses["confirmed"],
            "titles": len(titles),
            "images_saved": images_saved,
            "rejected_images_saved": rejected_saved,
        },
        "flagged_documents": flagged,
        "impact_if_all_removed": impact,
        "dev_after_current_exclusions": {
            "excluded": len(current),
            "unknown_ids": sorted(set(current) - dev_doc_ids),
            "datasets": after,
        },
        "page_pairs": pairs,
        "title_matches": [{"dev": d, "test": t, "similarity": r, "kind": k} for d, t, r, k in titles],
    }
    write_report(out_dir, report)
    print(f"\n{report['verdict']}\nReport: {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
