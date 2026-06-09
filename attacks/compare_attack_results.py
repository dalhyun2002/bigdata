from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ATTACK_DIR = os.path.join(PROJECT_DIR, "attacks", "Adversarial_Attacks")

DEFAULT_RUNS = {
    "baseline_efficientnetb0": "baseline_efficientnetb0_seed4",
    "proposed_efficientnetb0_se_mlp512_full": "proposed_efficientnetb0_se_mlp512_full_seed2",
}


def read_csv(path: str) -> List[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, rows: List[dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def collect_rows(attack_dir: str, runs: Dict[str, str]) -> List[dict]:
    rows: List[dict] = []
    for model_label, run_dir in runs.items():
        metrics_path = os.path.join(attack_dir, run_dir, "metrics_summary.csv")
        if not os.path.isfile(metrics_path):
            print(f"[skip] missing: {metrics_path}")
            continue
        for row in read_csv(metrics_path):
            rows.append(
                {
                    "model": model_label,
                    "run_dir": run_dir,
                    "method": row.get("method", ""),
                    "epsilon_px": row.get("epsilon_px", ""),
                    "n_test": row.get("n_test", ""),
                    "clean_accuracy": row.get("clean_accuracy", ""),
                    "adv_accuracy": row.get("adv_accuracy", ""),
                    "accuracy_drop": row.get("accuracy_drop", ""),
                    "attack_success_rate_clean_correct": row.get(
                        "attack_success_rate_clean_correct", ""
                    ),
                    "mean_layercam_mask_area": row.get("mean_layercam_mask_area", ""),
                    "mean_linf_px": row.get("mean_linf_px", ""),
                    "mean_psnr": row.get("mean_psnr", ""),
                    "mean_ssim": row.get("mean_ssim", ""),
                    "pgd_steps": row.get("pgd_steps", ""),
                    "alpha_px": row.get("alpha_px", ""),
                    "random_start": row.get("random_start", ""),
                    "layercam_threshold": row.get("layercam_threshold", ""),
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge baseline/proposed attack summaries.")
    parser.add_argument("--attack-dir", default=DEFAULT_ATTACK_DIR)
    parser.add_argument(
        "--out",
        default=os.path.join(DEFAULT_ATTACK_DIR, "attack_comparison_summary.csv"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = collect_rows(args.attack_dir, DEFAULT_RUNS)
    if not rows:
        raise FileNotFoundError("No attack metrics_summary.csv files were found.")
    write_csv(args.out, rows)
    print(f"Wrote {len(rows)} rows: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
