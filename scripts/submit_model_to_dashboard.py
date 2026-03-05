"""
Submit a trained model ZIP to the nnunet-dashboard.

Usage:
    python scripts/submit_model_to_dashboard.py <dataset_number> [zip_path]

Arguments:
    dataset_number  Numeric dataset ID (e.g. 289 for Dataset289_SRSBody)
    zip_path        Optional path to the model ZIP.
                    Defaults to <results_dir>/<Dataset###_Name>/model.zip

Example:
    python scripts/submit_model_to_dashboard.py 289
    python scripts/submit_model_to_dashboard.py 289 /path/to/model.zip

The dataset must already exist in the dashboard (same name, e.g. Dataset289_SRSBody).
"""
import sys
import re
from pathlib import Path

# Make sure the package is importable from the project root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nnunet_job_scheduler.config import config
from nnunet_job_scheduler import raw, ex


def find_dataset_id(dataset_num: str) -> str:
    num = str(int(dataset_num)).zfill(3)
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
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    dataset_num = sys.argv[1]
    dataset_id = find_dataset_id(dataset_num)
    configuration = "2d" if raw.is_2d(dataset_id) else "3d_lowres"

    # Resolve zip path
    if len(sys.argv) >= 3:
        zip_path = sys.argv[2]
    else:
        zip_path = str(Path(config["results_dir"]) / dataset_id / "model.zip")

    if not Path(zip_path).exists():
        raise SystemExit(
            f"ZIP not found: {zip_path}\n"
            "Run export_to_zip.py first, or pass the zip path as a second argument."
        )

    size_mb = Path(zip_path).stat().st_size / 1024 / 1024

    print(f"Dataset       : {dataset_id}")
    print(f"Configuration : {configuration}")
    print(f"ZIP path      : {zip_path}")
    print(f"ZIP size      : {size_mb:.1f} MB")
    print(f"Dashboard URL : {config.get('dashboard_url', '(not set)')}")
    print()
    print("Submitting to dashboard (chunked upload)...")

    model_id = ex.upload_to_dashboard(dataset_id, configuration, zip_path)
    ex._mark_uploaded(dataset_id, configuration, model_id)

    print()
    print(f"Model ID : {model_id}")
    print("Done. The model is now pending approval in the dashboard.")


if __name__ == "__main__":
    main()
