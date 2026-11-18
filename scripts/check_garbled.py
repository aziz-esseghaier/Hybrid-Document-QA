"""Measure the garbled-text rule: false alarms on real text, detection on corrupted text.

Usage:
    python scripts/check_garbled.py [--config configs/garbled.yaml] [--input results]

Reads the bridge tables from --input (default: RESULTS_ROOT), downloads the same PDFs (skipping any that changed) and
scores their text layers. Layers that agree with OCR count as clean; real garbled layers are rare, so detection is
measured by corrupting the clean pages in memory. French OCR text gives a false-alarm check for French.

Writes <RESULTS_ROOT>/garbled/<dataset>/: summary.json, pages.csv (ids and signals, no document text), config.yaml,
run_info.json.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import string
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable

import yaml

from doc_qa.core.config import get_settings, load_yaml
from doc_qa.core.parallel import read_jsonl, write_csv
from doc_qa.core.run_info import write_run_info
from doc_qa.ingest.pdf import clean_pdf_text, download_pdf, garbled_reasons, read_text_layers, text_signals
from doc_qa.ingest.text import clean_ocr_text

RULE_KEYS = {
    "bad_char_share": "max_bad_char_share",
    "symbol_share": "max_symbol_share",
    "spaced_letter_share": "max_spaced_letter_share",
    "long_word_share": "max_long_word_share",
    "common_word_share": "min_common_word_share",
}


def shift_codes(text: str, rng: random.Random) -> str:
    """Shift character codes by 29, as when a font has no Unicode map ("Contents" -> "&RQWHQWV").

    Args:
        text (str): Text.
        rng (random.Random): Unused.

    Returns:
        str: Corrupted text.
    """
    return "".join(chr(ord(c) - 29) if 62 <= ord(c) <= 126 else c for c in text)


def substitute_letters(text: str, rng: random.Random) -> str:
    """Replace each letter with another fixed letter, as with a scrambled font encoding.

    Args:
        text (str): Text.
        rng (random.Random): Random source.

    Returns:
        str: Corrupted text.
    """
    lower = string.ascii_lowercase
    mapped = "".join(rng.sample(lower, len(lower)))
    return text.translate(str.maketrans(lower + lower.upper(), mapped + mapped.upper()))


def replace_letters(text: str, rng: random.Random) -> str:
    """Turn 30% of the letters into the Unicode replacement character.

    Args:
        text (str): Text.
        rng (random.Random): Random source.

    Returns:
        str: Corrupted text.
    """
    return "".join("�" if c.isalpha() and rng.random() < 0.3 else c for c in text)


def private_use(text: str, rng: random.Random) -> str:
    """Map letters to private-use characters, as with symbol fonts.

    Args:
        text (str): Text.
        rng (random.Random): Unused.

    Returns:
        str: Corrupted text.
    """
    return "".join(chr(0xE000 + ord(c) % 0x1000) if c.isalpha() else c for c in text)


def space_letters(text: str, rng: random.Random) -> str:
    """Put a space between all characters of each line.

    Args:
        text (str): Text.
        rng (random.Random): Unused.

    Returns:
        str: Corrupted text.
    """
    return "\n".join(" ".join(line.replace(" ", "")) for line in text.split("\n"))


def drop_spaces(text: str, rng: random.Random) -> str:
    """Remove the spaces between words.

    Args:
        text (str): Text.
        rng (random.Random): Unused.

    Returns:
        str: Corrupted text.
    """
    return text.replace(" ", "")


CORRUPTIONS: dict[str, Callable[[str, random.Random], str]] = {
    "shifted_codes": shift_codes,
    "substituted_letters": substitute_letters,
    "replacement_chars": replace_letters,
    "private_use_chars": private_use,
    "spaced_letters": space_letters,
    "no_spaces": drop_spaces,
}


def flag_rates(groups: dict[str, list[dict[str, float]]], rule: dict[str, Any]) -> dict[str, float]:
    """Compute the share of pages a rule flags in each group.

    Args:
        groups (dict[str, list[dict[str, float]]]): Text signals per page, by group.
        rule (dict[str, Any]): Rule for `garbled_reasons`.

    Returns:
        dict[str, float]: Flagged share per group.
    """
    return {
        name: round(sum(bool(garbled_reasons(s, rule)) for s in pages) / len(pages), 4) if pages else 0.0
        for name, pages in groups.items()
    }


def sweep(groups: dict[str, list[dict[str, float]]], cfg: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Measure each check alone at several thresholds.

    Args:
        groups (dict[str, list[dict[str, float]]]): Text signals per page, by group.
        cfg (dict[str, Any]): Config with `sweeps` and `rule`.

    Returns:
        dict[str, list[dict[str, Any]]]: {signal: [{"threshold", "flagged": {group: share}}]}.
    """
    gates = {k: cfg["rule"][k] for k in ("min_letters", "min_distinct_words")}
    results: dict[str, list[dict[str, Any]]] = {}
    for signal, thresholds in cfg["sweeps"].items():
        results[signal] = []
        for threshold in thresholds:
            rule = {RULE_KEYS[signal]: threshold, **gates}
            results[signal].append({"threshold": threshold, "flagged": flag_rates(groups, rule)})
    return results


def read_layers(path: Path) -> list[str] | None:
    """Read a PDF's text layers (runs in a worker process).

    Args:
        path (Path): PDF file.

    Returns:
        list[str] | None: Raw text per page, or None if unreadable.
    """
    try:
        return read_text_layers(path)
    except Exception:  # PDFium raises its own error types for corrupt files
        return None


def load_layers(input_dir: Path, dataset: str, pdf_dir: Path, cfg: dict[str, Any]) -> tuple[dict[str, list[str]], int]:
    """Download the bridge PDFs and read their text layers.

    Args:
        input_dir (Path): Results folder of the bridge run.
        dataset (str): Dataset key.
        pdf_dir (Path): PDF folder.
        cfg (dict[str, Any]): Config.

    Returns:
        tuple[dict[str, list[str]], int]: Raw text per page by URL, and PDFs that were unavailable or changed.
    """
    with (input_dir / "bridge" / dataset / "pdfs.csv").open(encoding="utf-8", newline="") as f:
        pdfs = [r for r in csv.DictReader(f) if r["status"] == "ok"]
    usable = []
    for pdf in pdfs:
        path = pdf_dir / pdf["file"]
        if download_pdf(pdf["url"], path, cfg["download_timeout"]) != "ok":
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() == pdf["sha256"]:  # same file, so page numbers still hold
            usable.append((pdf["url"], path))
    with ProcessPoolExecutor(cfg["workers"]) as pool:
        texts = list(pool.map(read_layers, [path for _, path in usable]))
    layers = {url: pages for (url, _), pages in zip(usable, texts) if pages is not None}
    return layers, len(pdfs) - len(layers)


def main() -> None:
    """Score text layers and write the rule's results."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/garbled.yaml"))
    parser.add_argument("--input", type=Path, help="Results folder of the bridge run (default: RESULTS_ROOT).")
    args = parser.parse_args()

    settings = get_settings()
    cfg = load_yaml(args.config)
    dataset = cfg["dataset"]
    input_dir = args.input or settings.results_root
    results_dir = settings.results_root / "garbled" / dataset
    write_run_info(results_dir, seed=cfg["seed"], extra={"dataset": dataset})
    (results_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    pdf_dir = settings.data_root / "raw" / "pdfs" / cfg["urls_dataset"]
    layers, skipped_pdfs = load_layers(input_dir, dataset, pdf_dir, cfg)
    with (input_dir / "bridge" / dataset / "pages.csv").open(encoding="utf-8", newline="") as f:
        bridge_pages = list(csv.DictReader(f))

    rng = random.Random(cfg["seed"])
    groups: dict[str, list[dict[str, float]]] = {"clean_en": [], "low_agreement": [], "clean_fr_ocr": []}
    groups.update({name: [] for name in CORRUPTIONS})
    rows = []
    for page in bridge_pages:
        if page["fallback"] == "True" or page["url"] not in layers:
            continue
        raw = layers[page["url"]][int(page["pdf_page"]) - 1]
        signals = text_signals(raw)
        f1 = float(page["token_f1"])
        group = "clean_en" if f1 >= cfg["clean_min_f1"] else "low_agreement" if f1 < cfg["low_max_f1"] else "between"
        if group != "between":
            groups[group].append(signals)
        if group == "clean_en":
            text = clean_pdf_text(raw)
            for name, corrupt in CORRUPTIONS.items():
                groups[name].append(text_signals(corrupt(text, rng)))
        reasons = garbled_reasons(signals, cfg["rule"])
        rows.append({"page_id": page["page_id"], "url": page["url"], "pdf_page": page["pdf_page"], "group": group,
                     "token_f1": f1, **signals, "reasons": " ".join(reasons)})

    for french in cfg["french_ocr"]:
        shards = sorted((settings.data_root / "derived" / "ocr" / french).glob("shard*.jsonl"))
        if not shards:
            raise SystemExit(f"No OCR output for {french} (run ocr_store.py pull first).")
        for record in read_jsonl(shards):
            signals = text_signals(clean_ocr_text(record["text"]).replace(" | ", " "))  # table separators are OCR's own
            if signals["word_chars"] >= cfg["min_word_chars"]:
                groups["clean_fr_ocr"].append(signals)

    summary = {
        "pdfs_skipped": skipped_pdfs,
        "pages": {name: len(pages) for name, pages in groups.items()},
        "rule": cfg["rule"],
        "flagged_by_rule": flag_rates(groups, cfg["rule"]),
        "sweeps": sweep(groups, cfg),
    }
    write_csv(results_dir / "pages.csv", rows)
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"PDFs skipped: {skipped_pdfs}")
    print(f"{'group':<22} {'pages':>6} {'flagged':>8}")
    for name, rate in summary["flagged_by_rule"].items():
        print(f"{name:<22} {len(groups[name]):>6} {rate:>8.3f}")


if __name__ == "__main__":
    main()
