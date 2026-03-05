"""
Check whether all training folds are complete for a given dataset number.

Usage:
    python scripts/tr_check_complete.py <dataset_number>

Example:
    python scripts/tr_check_complete.py 289
"""
import sys
import json
import re
from pathlib import Path

# Make sure the package is importable from the project root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nnunet_job_scheduler.config import config
from nnunet_job_scheduler import tr, raw


def find_dataset_id(dataset_num: str) -> str:
    """Find the full Dataset###_Name string for a given numeric ID."""
    num = str(int(dataset_num)).zfill(3)  # normalise to 3 digits
    pattern = re.compile(rf"^Dataset{num}_")
    results_dir = Path(config["results_dir"])
    raw_dir = Path(config["raw_dir"])

    for search_dir in (results_dir, raw_dir):
        if search_dir.exists():
            for entry in search_dir.iterdir():
                if entry.is_dir() and pattern.match(entry.name):
                    return entry.name

    raise SystemExit(f"No dataset found for number {dataset_num} in results or raw directories.")


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    dataset_num = sys.argv[1]
    dataset_id = find_dataset_id(dataset_num)
    configuration = "2d" if raw.is_2d(dataset_id) else "3d_lowres"

    print(f"Dataset : {dataset_id}")
    print(f"Config  : {configuration}")
    print()

    s = tr.status(dataset_id)

    # Per-fold summary
    folds = [0, 1, 2, 3, 4]
    all_done = True
    for fold in folds:
        chk_best  = tr.checkpoint_best_exists_for_config(dataset_id, fold, configuration)
        chk_final = tr.checkpoint_final_exists(dataset_id, fold)
        val_done  = tr.validation_summary_file_exists(dataset_id, fold)
        fold_done = chk_best["exists"] and chk_final["exists"] and val_done["exists"]
        if not fold_done:
            all_done = False
        status_str = "DONE" if fold_done else "INCOMPLETE"
        print(
            f"  fold {fold}: {status_str}"
            f"  | checkpoint_best={chk_best['exists']}"
            f"  checkpoint_final={chk_final['exists']}"
            f"  validation={val_done['exists']}"
        )

    print()
    if all_done:
        print("Result: ALL FOLDS COMPLETE")
    else:
        print("Result: TRAINING INCOMPLETE")
        print()
        print("Full status:")
        print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
