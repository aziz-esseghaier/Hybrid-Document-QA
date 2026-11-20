"""Merge the visual and text page rankings with weighted reciprocal rank fusion, and choose the weights on dev.

Usage:
    python scripts/eval_fusion.py [--config configs/fusion.yaml]

Runs on saved rankings and judgments (export_qrels.py); no GPU or Qdrant. Scores every weight combination in the grid
by its mean NDCG@10 over the dataset-language groups and keeps the best. Checks how well that choice carries over by
leaving each dataset out in turn, and compares the best setting with each branch alone (bootstrap intervals;
translations of a question are resampled together). Writes <RESULTS_ROOT>/<results_name>/: summary.json,
grid.jsonl, config.yaml, run_info.json, and <dataset>/rankings_fused.jsonl. configs/fusion_rerank.yaml scores the
reranker the same way, as one more ranking; `restrict` keeps only the reranked pages within visual's top k.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

from doc_qa.core.config import REPO_ROOT, get_settings, load_yaml
from doc_qa.core.parallel import read_jsonl, write_jsonl
from doc_qa.core.run_info import write_run_info
from doc_qa.evaluation.metrics import cluster_bootstrap_ci, evaluate, ndcg_at_k
from doc_qa.retrieval.fusion import rrf

LANGUAGE_RULE = "bm25_same_language_only"


def load_queries(cfg: dict[str, Any], specs: dict[str, Any]) -> list[dict[str, Any]]:
    """Load every dev query with its judgments and saved rankings.

    Args:
        cfg (dict[str, Any]): Fusion config.
        specs (dict[str, Any]): Dataset specs.

    Returns:
        list[dict[str, Any]]: {dataset, query_id, group, cross_language, relevance, lists, cluster} per query.
    """
    queries = []
    for dataset in cfg["datasets"]:
        rows = read_jsonl([REPO_ROOT / cfg["qrels"].format(dataset=dataset)])
        if not rows:
            raise SystemExit(f"No judgments for {dataset} (run export_qrels.py).")
        lists = {}
        for source, pattern in cfg["rankings"].items():
            saved = read_jsonl([REPO_ROOT / pattern.format(dataset=dataset)])
            lists[source] = {row["query_id"]: [page for page, _ in row["pages"]] for row in saved}
        for row in rows:
            relevant = tuple(sorted(page for page, grade in row["pages"].items() if grade > 0))
            ranked = {source: by_query.get(row["query_id"], []) for source, by_query in lists.items()}
            queries.append({
                "dataset": dataset,
                "query_id": row["query_id"],
                "group": f"{dataset}/{row['language']}",
                "cross_language": row["language"] != specs[dataset]["doc_language"],
                "relevance": row["pages"],
                "lists": restrict(ranked, cfg.get("restrict", {})),
                "cluster": (dataset, relevant),  # translations share their relevant pages
            })
    return queries


def restrict(lists: dict[str, list[str]], rules: dict[str, Sequence[Any]]) -> dict[str, list[str]]:
    """Keep only the pages of a ranking that another ranking puts in its top k.

    Args:
        lists (dict[str, list[str]]): Ranked page ids per source, best first.
        rules (dict[str, Sequence[Any]]): {source: [other source, k]}, e.g. {"rerank": ["visual", 20]}.

    Returns:
        dict[str, list[str]]: The lists, with each restricted source cut down in its own order.
    """
    kept = dict(lists)
    for source, (other, k) in rules.items():
        allowed = set(lists[other][:k])
        kept[source] = [page for page in lists[source] if page in allowed]
    return kept


def grid_settings(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """List the weight combinations worth scoring.

    Args:
        grid (dict[str, list[Any]]): Values per ranking weight, and optionally of the BM25 language rule.

    Returns:
        list[dict[str, Any]]: Settings; visual alone and rules that change nothing are left out.
    """
    sources = [key for key in grid if key != LANGUAGE_RULE]
    settings = []
    for combo in itertools.product(*(grid[source] for source in sources), grid.get(LANGUAGE_RULE, [False])):
        setting = {**dict(zip(sources, combo[:-1])), LANGUAGE_RULE: combo[-1]}
        if not any(setting[source] for source in sources if source != "visual"):
            continue
        if setting[LANGUAGE_RULE] and not setting.get("bm25"):
            continue
        settings.append(setting)
    return settings


def label(setting: dict[str, Any]) -> str:
    """Name a setting for tables.

    Args:
        setting (dict[str, Any]): Weights, and optionally the BM25 language rule.

    Returns:
        str: E.g. "visual 2 + dense 1 + bm25 0.5 (same language)".
    """
    parts = [f"{source} {weight:g}" for source, weight in setting.items() if source != LANGUAGE_RULE and weight]
    return " + ".join(parts) + (" (same language)" if setting.get(LANGUAGE_RULE) else "")


def fuse(query: dict[str, Any], setting: dict[str, Any], k: int, limit: int) -> list[tuple[str, float]]:
    """Fuse one query's saved rankings.

    Args:
        query (dict[str, Any]): Query from `load_queries`.
        setting (dict[str, Any]): Weight per ranking (missing = 0), and optionally the BM25 language rule.
        k (int): RRF damping constant.
        limit (int): Pages kept.

    Returns:
        list[tuple[str, float]]: (page id, fused score), best first.
    """
    drop_bm25 = query["cross_language"] and setting.get(LANGUAGE_RULE)
    weights = [0 if drop_bm25 and source == "bm25" else setting.get(source, 0) for source in query["lists"]]
    return rrf(list(query["lists"].values()), k, limit, weights)


def ndcg_values(queries: list[dict[str, Any]], setting: dict[str, Any], k: int, limit: int) -> np.ndarray:
    """Score one setting on every query.

    Args:
        queries (list[dict[str, Any]]): Queries from `load_queries`.
        setting (dict[str, Any]): Fusion setting.
        k (int): RRF damping constant.
        limit (int): Pages kept.

    Returns:
        np.ndarray: NDCG@10 per query.
    """
    return np.array([
        ndcg_at_k([page for page, _ in fuse(query, setting, k, limit)], query["relevance"]) for query in queries
    ])


def group_means(queries: list[dict[str, Any]], values: np.ndarray) -> dict[str, float]:
    """Average per-query values within each dataset-language group.

    Args:
        queries (list[dict[str, Any]]): Queries.
        values (np.ndarray): One value per query.

    Returns:
        dict[str, float]: Mean per group.
    """
    by_group: dict[str, list[float]] = defaultdict(list)
    for query, value in zip(queries, values):
        by_group[query["group"]].append(float(value))
    return {group: float(np.mean(group_values)) for group, group_values in sorted(by_group.items())}


def macro(means: dict[str, float], include: Sequence[str] | None = None, exclude: str | None = None) -> float:
    """Average group means, optionally over some datasets only.

    Args:
        means (dict[str, float]): Mean per "<dataset>/<language>" group.
        include (Sequence[str] | None): Datasets to keep; None = all.
        exclude (str | None): Dataset to leave out.

    Returns:
        float: Mean over the kept groups.
    """
    kept = [
        value for group, value in means.items()
        if (include is None or group.split("/")[0] in include) and group.split("/")[0] != exclude
    ]
    return float(np.mean(kept))


def leave_one_dataset_out(
    settings: list[dict[str, Any]], means: list[dict[str, float]], reference: dict[str, float], datasets: Sequence[str]
) -> dict[str, dict[str, Any]]:
    """For each dataset, choose the setting on the other datasets and score it on this one.

    Args:
        settings (list[dict[str, Any]]): Grid settings.
        means (list[dict[str, float]]): Group means per setting, in the same order.
        reference (dict[str, float]): Group means of a baseline, reported alongside.
        datasets (Sequence[str]): Datasets.

    Returns:
        dict[str, dict[str, Any]]: {dataset: {setting, ndcg@10, reference}}.
    """
    result = {}
    for held_out in datasets:
        best = max(range(len(settings)), key=lambda i: macro(means[i], exclude=held_out))
        result[held_out] = {
            "setting": label(settings[best]),
            "ndcg@10": round(macro(means[best], include=[held_out]), 4),
            "reference": round(macro(reference, include=[held_out]), 4),
        }
    return result


def compare(
    queries: list[dict[str, Any]], first: np.ndarray, second: np.ndarray, resamples: int, seed: int
) -> dict[str, dict[str, float]]:
    """Mean difference between two settings with a bootstrap interval, per group and over all queries.

    Args:
        queries (list[dict[str, Any]]): Queries.
        first (np.ndarray): NDCG@10 per query of the first setting.
        second (np.ndarray): The same for the second setting.
        resamples (int): Bootstrap samples.
        seed (int): Random seed.

    Returns:
        dict[str, dict[str, float]]: {group or "all": {queries, delta, low, high}}.
    """
    deltas = first - second
    groups = {"all": list(range(len(queries)))}
    for i, query in enumerate(queries):
        groups.setdefault(query["group"], []).append(i)
    result = {}
    for group, index in groups.items():
        clusters = [queries[i]["cluster"] for i in index]
        mean, low, high = cluster_bootstrap_ci(deltas[index], clusters, resamples, seed=seed)
        result[group] = {"queries": len(index), "delta": round(mean, 4), "low": round(low, 4), "high": round(high, 4)}
    return result


def saved_metrics_match(source: Sequence[str], dataset: str, computed: dict[str, dict[str, float]]) -> bool:
    """Check that a baseline recomputed from saved rankings gives the scores saved by its own evaluation.

    Args:
        source (Sequence[str]): Metrics file pattern and the method in it, e.g. [".../metrics.json", "dense"].
        dataset (str): Dataset key.
        computed (dict[str, dict[str, float]]): Its `evaluate` output.

    Returns:
        bool: True if NDCG@10 and Recall@100 match for every group.
    """
    path, method = source
    saved = json.loads((REPO_ROOT / path.format(dataset=dataset)).read_text(encoding="utf-8"))[method]
    return all(
        computed[group][metric] == values[metric]
        for group, values in saved.items()
        for metric in ("ndcg@10", "recall@100")
    )


def main() -> None:
    """Score the grid, choose the best setting, and compare it with the baselines."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/fusion.yaml"))
    args = parser.parse_args()

    settings = get_settings()
    cfg = load_yaml(args.config)
    specs = load_yaml(Path(cfg["datasets_config"]))["datasets"]
    k, limit = cfg["rrf_k"], cfg["page_limit"]
    queries = load_queries(cfg, specs)
    results_dir = settings.results_root / cfg["results_name"]
    write_run_info(results_dir, seed=cfg["seed"], extra={"datasets": cfg["datasets"], "queries": len(queries)})
    (results_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    grid = grid_settings(cfg["grid"])
    grid_values = [ndcg_values(queries, setting, k, limit) for setting in grid]
    grid_means = [group_means(queries, values) for values in grid_values]
    write_jsonl(results_dir / "grid.jsonl", [
        {
            "setting": setting,
            "macro_ndcg@10": round(macro(means), 4),
            "ndcg@10": {group: round(value, 4) for group, value in means.items()},
        }
        for setting, means in zip(grid, grid_means)
    ])
    best_index = max(range(len(grid)), key=lambda i: macro(grid_means[i]))
    best = grid[best_index]

    methods = {name: dict(weights) for name, weights in cfg["baselines"].items()}
    methods["best"] = best
    values = {name: ndcg_values(queries, setting, k, limit) for name, setting in methods.items()}
    means = {name: group_means(queries, method_values) for name, method_values in values.items()}
    summary: dict[str, Any] = {
        "selection": "mean NDCG@10 over dataset-language groups",
        "best": {"setting": best, "label": label(best)},
        "settings_scored": len(grid),
        "macro_ndcg@10": {name: round(macro(group), 4) for name, group in means.items()},
        "ndcg@10": {name: {g: round(m, 4) for g, m in group.items()} for name, group in means.items()},
        "leave_one_dataset_out": leave_one_dataset_out(grid, grid_means, means["visual"], cfg["datasets"]),
        "best_minus": {
            name: compare(queries, values["best"], values[name], cfg["bootstrap_resamples"], cfg["seed"])
            for name in cfg["compare_with"]
        },
    }

    metrics: dict[str, dict[str, Any]] = defaultdict(dict)
    matches = []
    for dataset in cfg["datasets"]:
        subset = [query for query in queries if query["dataset"] == dataset]
        qrels = {query["query_id"]: query["relevance"] for query in subset}
        languages = {query["query_id"]: query["group"].split("/")[1] for query in subset}
        for name, setting in methods.items():
            fused = {query["query_id"]: fuse(query, setting, k, limit) for query in subset}
            ranked = {query_id: [page for page, _ in pages] for query_id, pages in fused.items()}
            metrics[name][dataset] = evaluate(ranked, qrels, languages, cfg["recall_ks"])
            if name in cfg["saved_metrics"]:
                matches.append(saved_metrics_match(cfg["saved_metrics"][name], dataset, metrics[name][dataset]))
            if name == "best":
                write_jsonl(results_dir / dataset / "rankings_fused.jsonl", [
                    {"query_id": query_id, "pages": [[page, round(score, 6)] for page, score in pages]}
                    for query_id, pages in fused.items()
                ])
    summary["baselines_match_saved_metrics"] = all(matches)
    summary["metrics"] = metrics
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Best: {label(best)} ({len(grid)} settings scored)\n")
    print(f"{'NDCG@10':<28}" + "".join(f"{name:>10}" for name in methods))
    for group in means["visual"]:
        print(f"{group:<28}" + "".join(f"{means[name][group]:>10.3f}" for name in methods))
    print(f"{'mean':<28}" + "".join(f"{summary['macro_ndcg@10'][name]:>10.3f}" for name in methods) + "\n")
    for name, by_group in summary["best_minus"].items():
        overall = by_group["all"]
        print(f"best - {name}: {overall['delta']:+.3f} [{overall['low']:+.3f}, {overall['high']:+.3f}] over all queries")
    print("\nLeave one dataset out (setting chosen on the others; visual alone for reference):")
    for dataset, row in summary["leave_one_dataset_out"].items():
        print(f"  {dataset:<26} {row['ndcg@10']:.3f} vs {row['reference']:.3f}  {row['setting']}")
    if not summary["baselines_match_saved_metrics"]:
        print("WARNING: recomputed baselines differ from the saved metrics; check the judgments.")


if __name__ == "__main__":
    main()
