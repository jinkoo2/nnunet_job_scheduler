"""
Export a trained nnUNet model to a ZIP file.

Usage:
    python scripts/export_to_zip.py <dataset_number>

Example:
    python scripts/export_to_zip.py 289

The ZIP is written to:
    <results_dir>/<Dataset###_Name>/model.zip
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
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    dataset_num = sys.argv[1]
    dataset_id = find_dataset_id(dataset_num)
    configuration = "2d" if raw.is_2d(dataset_id) else "3d_lowres"

    print(f"Dataset       : {dataset_id}")
    print(f"Configuration : {configuration}")
    print(f"Results dir   : {config['results_dir']}")
    print()
    print("Exporting model...")

    zip_path = ex.export_model(dataset_id, configuration)

    size_bytes = Path(zip_path).stat().st_size
    size_mb = size_bytes / 1024 / 1024
    size_gb = size_mb / 1024

    print()
    print(f"ZIP path : {zip_path}")
    print(f"Size     : {size_mb:.1f} MB  ({size_gb:.2f} GB)")
    print()
    print("Export complete.")


if __name__ == "__main__":
    main()
