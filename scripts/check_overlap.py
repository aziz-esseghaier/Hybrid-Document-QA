"""Check the dev datasets (ViDoRe V1/V2) against the test set (ViDoRe V3) for overlap.

Loads every dataset, hashes every page image, compares dev with test (exact and perceptual page hashes,
document titles and URLs), and reports which dev documents to remove and what removing them costs.

Usage:
    python scripts/check_overlap.py --config configs/m0_overlap.yaml

Outputs, in <RESULTS_ROOT>/<config name>/:
    config.yaml, run_info.json    what was run
    report.md, report.json        the results
    proposed_exclusions.yaml      flagged dev documents; review them, then copy the confirmed ids
                                  into configs/dev_exclusions.yaml
    artifacts/pairs/*.png         matched pages side by side (dev left, test right)
"""

from __future__ import annotations

import argparse
import json
import re
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
    save_hashes,
    side_by_side,
)
from doc_qa.evaluation.vidore import apply_exclusions, iter_page_images, load_benchmarks


def page_hashes(bench: Benchmark, revision: str, cache_dir: Path, workers: int | None) -> list[PageHash]:
    """Hash every page of a dataset, reusing cached hashes when they cover all its pages.

    The cache file name contains the dataset revision, so new data is never matched with old hashes.

    Args:
        bench (Benchmark): The dataset.
        revision (str): The dataset's pinned Hub revision.
        cache_dir (Path): Folder for the cached hash files.
        workers (int | None): Hashing processes; None uses all CPU cores.

    Returns:
        list[PageHash]: One hash per page.
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
    """Count a dataset's documents, pages, queries and judgments.

    Args:
        bench (Benchmark): The dataset.

    Returns:
        dict[str, Any]: The counts, plus loading warnings.
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
    """Merge exact and perceptual matches into one list of page pairs, most certain first.

    Args:
        exact (list[tuple[str, str]]): (dev page, test page) pairs with identical pixels.
        near (list[tuple[str, str, int]]): (dev page, test page, pHash distance) pairs.

    Returns:
        list[dict[str, Any]]: One dict per pair: dev_page, test_page, exact, phash_distance.
    """
    exact_set = set(exact)
    distance = {(d, t): dist for d, t, dist in near}
    pairs = [
        {"dev_page": d, "test_page": t, "exact": (d, t) in exact_set, "phash_distance": distance.get((d, t))}
        for d, t in exact_set | set(distance)
    ]
    return sorted(pairs, key=lambda p: (not p["exact"], p["phash_distance"] or 0, p["dev_page"], p["test_page"]))


def flag_documents(
    pairs: list[dict[str, Any]], titles: list[tuple[str, str, float, str]], page_doc: dict[str, str]
) -> dict[str, dict[str, Any]]:
    """Group the evidence by dev document. Every dev document with any evidence is flagged.

    Args:
        pairs (list[dict[str, Any]]): Matched page pairs from `collect_page_pairs`.
        titles (list[tuple[str, str, float, str]]): Matched documents from `find_title_matches`.
        page_doc (dict[str, str]): Document id of every page id.

    Returns:
        dict[str, dict[str, Any]]: Evidence per flagged dev document id, sorted by id.
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
    """Measure what each dev dataset loses if the given documents are removed.

    Args:
        dev (list[Benchmark]): Dev datasets.
        excluded (list[str]): Document ids to remove.
        large_share (float): Share of lost queries above which the overlap counts as large.

    Returns:
        list[dict[str, Any]]: Per dataset: documents, pages and queries before and after, share lost, large flag.
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
    """Turn a page id into a short, file-system-safe string.

    Args:
        page_id (str): The page id.
        max_len (int): Maximum length kept.

    Returns:
        str: The safe string.
    """
    return re.sub(r"[^\w.-]+", "_", page_id)[-max_len:]


def save_pair_images(pairs: list[dict[str, Any]], benches: list[Benchmark], out_dir: Path, limit: int) -> None:
    """Save the first `limit` page pairs side by side, and store each image's path in its pair dict.

    Args:
        pairs (list[dict[str, Any]]): Matched page pairs, most certain first.
        benches (list[Benchmark]): All datasets, to read the images from.
        out_dir (Path): The experiment's results folder.
        limit (int): Maximum number of images.
    """
    chosen = pairs[:limit]
    by_name = {b.name: b for b in benches}
    needed: dict[str, set[str]] = defaultdict(set)
    for p in chosen:
        for page_id in (p["dev_page"], p["test_page"]):
            needed[page_id.split("/", 1)[0]].add(page_id)
    images: dict[str, bytes] = {}
    for name, page_ids in needed.items():
        images.update(iter_page_images(by_name[name], page_ids))

    pair_dir = out_dir / "artifacts" / "pairs"
    pair_dir.mkdir(parents=True, exist_ok=True)
    for n, p in enumerate(chosen):
        tag = "exact" if p["exact"] else f"d{p['phash_distance']:02d}"
        file_name = f"{n:03d}_{tag}_{_safe_name(p['dev_page'])}__{_safe_name(p['test_page'])}.png"
        side_by_side(images[p["dev_page"]], images[p["test_page"]]).save(pair_dir / file_name)
        p["image"] = f"artifacts/pairs/{file_name}"


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    """Format a Markdown table.

    Args:
        headers (list[str]): Column titles.
        rows (list[list[Any]]): Cell values, one list per row.

    Returns:
        str: The table.
    """
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def verdict(flagged: dict[str, Any], impact: list[dict[str, Any]]) -> str:
    """Summarize the outcome in one sentence.

    Args:
        flagged (dict[str, Any]): Flagged dev documents.
        impact (list[dict[str, Any]]): Removal impact per dev dataset.

    Returns:
        str: The verdict.
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
        out_dir (Path): The experiment's results folder.
        report (dict[str, Any]): Everything the check found.
    """
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    proposed = {"ids": list(report["flagged_documents"]), "evidence": report["flagged_documents"]}
    with (out_dir / "proposed_exclusions.yaml").open("w", encoding="utf-8") as f:
        f.write("# Flagged dev documents. Check each one in artifacts/pairs/, then copy the confirmed ids\n")
        f.write("# into configs/dev_exclusions.yaml.\n")
        yaml.safe_dump(proposed, f, sort_keys=False, allow_unicode=True)

    m = report["matches"]
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
        f"- Exact page matches: {m['exact_pages']}",
        f"- Near-duplicate pages (pHash distance ≤ {report['config']['phash_max_distance']}): {m['near_pages']}",
        f"- Title or URL matches: {m['titles']}",
        f"- Side-by-side images saved: {m['images_saved']} (artifacts/pairs/)",
        "",
        "## Flagged dev documents",
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
    """Run the overlap check and write the report."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/m0_overlap.yaml"))
    args = parser.parse_args()

    settings = get_settings()
    cfg = load_yaml(args.config)
    specs = load_yaml(Path(cfg["datasets_config"]))["datasets"]
    out_dir = settings.results_root / cfg["name"]
    write_run_info(out_dir, extra={"datasets": {k: {"repo": s["repo"], "revision": s["revision"]} for k, s in specs.items()}})
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    # 1. Load every dataset without exclusions: the check must see the raw dev set.
    benches = load_benchmarks(specs, settings.data_root / "raw", cfg["languages"])
    dev = [b for b in benches if b.role == "dev"]
    test = [b for b in benches if b.role == "test"]
    summaries = [summarize(b) for b in benches]
    for s in summaries:
        print(f"{s['dataset']:<26} {s['role']:<4} pages={s['pages']:>5} queries={s['queries_en'] + s['queries_fr']:>5}")

    # 2. Hash every page (cached per dataset revision).
    cache_dir = settings.data_root / "derived" / "page_hashes"
    hashes = {b.name: page_hashes(b, specs[b.name]["revision"], cache_dir, cfg["workers"]) for b in benches}
    dev_hashes = [h for b in dev for h in hashes[b.name]]
    test_hashes = [h for b in test for h in hashes[b.name]]

    # 3. Compare dev with test.
    exact = find_exact_matches(dev_hashes, test_hashes)
    near = find_phash_matches(dev_hashes, test_hashes, cfg["phash_max_distance"], cfg["max_aspect_ratio_diff"])
    titles = find_title_matches(
        [d for b in dev for d in b.documents.values()],
        [d for b in test for d in b.documents.values()],
        cfg["title_min_ratio"],
    )
    pairs = collect_page_pairs(exact, near)
    page_doc = {pid: p.doc_id for b in benches for pid, p in b.pages.items()}
    flagged = flag_documents(pairs, titles, page_doc)
    impact = removal_impact(dev, list(flagged), cfg["large_overlap_query_share"])
    save_pair_images(pairs, benches, out_dir, cfg["max_pair_images"])

    # 4. Dev as the loader will serve it, with the reviewed exclusions applied.
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
            "titles": len(titles),
            "images_saved": min(len(pairs), cfg["max_pair_images"]),
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
