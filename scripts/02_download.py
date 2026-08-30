"""Download one ego and one exo view for each selected recording.

Source is the HuggingFace mirror `cvml-nus/assembly101` (gated: auto -- accept
the terms once on the dataset page; the local HF token is reused).

Only two of the twelve available views are fetched per recording. That matters:
exo views run 500MB-1GB each while ego views are ~30MB, so view selection is the
entire download budget.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download, get_hf_file_metadata, hf_hub_url

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from exoego.paths import assembly_root, ensure_dirs, manifests_dir, recordings_dir

REPO_ID = "cvml-nus/assembly101"


def adopt_existing_sample(wanted) -> int:
    """Symlink any already-downloaded videos under sample/ into recordings/.

    Avoids re-fetching the ~720MB recording that was downloaded by hand.
    """
    sample_root = assembly_root() / "sample" / "recordings"
    if not sample_root.exists():
        return 0

    adopted = 0
    for seq, view in wanted:
        source = sample_root / seq / f"{view}.mp4"
        target = recordings_dir() / seq / f"{view}.mp4"
        if not source.exists() or target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source.resolve())
        adopted += 1
        print(f"  adopted existing {seq}/{view}.mp4")
    return adopted


def remote_size(filename: str) -> int:
    url = hf_hub_url(repo_id=REPO_ID, filename=filename, repo_type="dataset")
    return get_hf_file_metadata(url).size or 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yes", action="store_true", help="skip the size confirmation")
    parser.add_argument("--dry-run", action="store_true", help="report sizes and exit")
    args = parser.parse_args()

    ensure_dirs()
    recordings = pd.read_csv(manifests_dir() / "recordings.csv")

    wanted = []
    for row in recordings.itertuples(index=False):
        wanted.append((row.seq, row.ego_view))
        wanted.append((row.seq, row.exo_view))

    print(f"{len(recordings)} recordings, {len(wanted)} video files requested")
    adopt_existing_sample(wanted)

    pending = []
    for seq, view in wanted:
        target = recordings_dir() / seq / f"{view}.mp4"
        if target.exists() and target.stat().st_size > 0:
            continue
        pending.append((seq, view))

    if not pending:
        print("everything already present -- nothing to download")
        return

    print(f"{len(pending)} files still to fetch; querying sizes ...", flush=True)
    total_bytes = 0
    for seq, view in pending:
        total_bytes += remote_size(f"recordings/{seq}/{view}.mp4")
    print(f"total download: {total_bytes / 1e9:.1f} GB")

    if args.dry_run:
        return
    if not args.yes:
        print("re-run with --yes to start the download")
        return

    for index, (seq, view) in enumerate(pending, start=1):
        print(f"[{index}/{len(pending)}] {seq}/{view}.mp4", flush=True)
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=f"recordings/{seq}/{view}.mp4",
            local_dir=str(assembly_root()),
        )

    print("done")


if __name__ == "__main__":
    main()
