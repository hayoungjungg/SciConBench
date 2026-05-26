"""Download SciConBench evaluation data from Google Drive for reproducing
the analysis/results in the paper and support future research. 

This downloads the SciConBench data archives from a public Google Drive folder,
unzips them, and places them in one central repository-level data directory.

Files downloaded:
- ``labeled_facts.zip``: Contains the facts labeled by LLM judge for measuring 
  factual precision and recall. For reproducing the analysis/results in the paper.
- ``llm-judge-human-annotations.zip``: Contains theexpert annotations over atomic facts, 
  used to align and validate the LLM judge for measuring factual precision and recall.
- ``model_response.zip``: Contains the generated conclusions across the evaluated models.
  For understanding model behavior and supporting future downstream analysis.
- ``preprocessed_facts.zip``: Contains the decomposed facts extracted from the generated
  conclusions.
"""

from __future__ import annotations

import argparse
import importlib
import shutil
import sys
import zipfile
from pathlib import Path


DRIVE_FOLDER_URL = (
    "https://drive.google.com/drive/folders/1vDv9ZpM1mKAacKzhOMEZwOn805rzmRyN"
)
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data"
EXPECTED_ARCHIVES = (
    "labeled_facts.zip",
    "llm-judge-human-annotations.zip",
    "model_response.zip",
    "preprocessed_facts.zip",
)


def _load_gdown():
    try:
        return importlib.import_module("gdown")
    except ImportError as exc:  # pragma: no cover - exercised only when dependency is missing
        raise SystemExit(
            "Missing dependency: gdown. Install project dependencies with "
            "`pip install -r requirements.txt` or install this package before "
            "running the downloader."
        ) from exc


def _is_macos_metadata(name: str) -> bool:
    """Skip macOS Finder/Archive Utility metadata that ships inside the zips."""
    if name.startswith("__MACOSX/") or "/__MACOSX/" in name:
        return True
    basename = name.rsplit("/", 1)[-1]
    if basename == ".DS_Store":
        return True
    if basename.startswith("._"):
        return True
    return False


def _extract_archive(archive_path: Path, data_dir: Path, keep_archive: bool) -> None:
    target_dir = data_dir / archive_path.stem
    if target_dir.exists():
        print(f"[Already exists] Skipping extraction: {target_dir}")
        if not keep_archive:
            archive_path.unlink(missing_ok=True)
        return

    print(f"[Extracting] {archive_path.name} -> {target_dir}")
    data_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        members = [m for m in archive.infolist() if not _is_macos_metadata(m.filename)]
        for member in members:
            archive.extract(member, data_dir)

    if keep_archive:
        print(f"[Kept archive] {archive_path}")
    else:
        archive_path.unlink()


def download_data(
    data_dir: Path,
    folder_url: str,
    unzip: bool,
    keep_archives: bool,
    force: bool,
) -> None:
    gdown = _load_gdown()

    data_dir.mkdir(parents=True, exist_ok=True)

    if force:
        for archive_name in EXPECTED_ARCHIVES:
            archive_path = data_dir / archive_name
            extracted_dir = data_dir / Path(archive_name).stem
            archive_path.unlink(missing_ok=True)
            if extracted_dir.exists():
                shutil.rmtree(extracted_dir)

    print(f"[Download location] {data_dir}")
    gdown.download_folder(
        url=folder_url,
        output=str(data_dir),
        quiet=False,
        use_cookies=False,
        resume=True,
    )

    downloaded_archives = sorted(data_dir.glob("*.zip"))
    downloaded_names = {path.name for path in downloaded_archives}
    missing = sorted(set(EXPECTED_ARCHIVES) - downloaded_names)
    if missing:
        print(
            "[Warning] Expected archive(s) not found after download: "
            + ", ".join(missing)
        )

    if unzip:
        for archive_path in downloaded_archives:
            _extract_archive(archive_path, data_dir, keep_archives)

    print("[Done] SciConBench data is available in:", data_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download SciConBench data archives from Google Drive."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Central data directory. Defaults to {DEFAULT_DATA_DIR}.",
    )
    parser.add_argument(
        "--folder-url",
        default=DRIVE_FOLDER_URL,
        help="Public Google Drive folder URL containing the data archives.",
    )
    parser.add_argument(
        "--no-unzip",
        action="store_true",
        help="Only download zip files; do not extract them.",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep zip files after extraction.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove existing expected archives/extracted folders before downloading.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    download_data(
        data_dir=args.data_dir.expanduser().resolve(),
        folder_url=args.folder_url,
        unzip=not args.no_unzip,
        keep_archives=args.keep_archives,
        force=args.force,
    )


if __name__ == "__main__":
    try:
        main()
    except zipfile.BadZipFile as exc:
        raise SystemExit(f"Failed to extract zip archive: {exc}") from exc
    except KeyboardInterrupt:
        sys.exit("Interrupted.")
