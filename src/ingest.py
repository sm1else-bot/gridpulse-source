"""
EIA Form 930 bulk CSV downloader.
Downloads hourly grid balance data for all balancing authorities, 2019-2024.
Files arrive as semi-annual chunks (~200-400 MB each, 12 files per year).
"""

import sys
from pathlib import Path

import requests
from tqdm import tqdm

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
BASE_URL = "https://www.eia.gov/electricity/gridmonitor/sixMonthFiles"
YEARS = range(2019, 2025)
HALVES = ["Jan_Jun", "Jul_Dec"]


_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.eia.gov/electricity/gridmonitor/about",
}
_MIN_VALID_BYTES = 1 * 1024 * 1024  # anything under 1 MB is an error page


def _download_one(url: str, dest: Path, session: requests.Session) -> str:
    """Download a single file with an inner tqdm bar. Returns status string."""
    resp = session.get(url, stream=True, timeout=120, headers=_HEADERS)
    if resp.status_code == 404:
        return "missing"
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    dest.parent.mkdir(parents=True, exist_ok=True)

    with open(dest, "wb") as fh, tqdm(
        desc=f"  {dest.name}",
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        leave=False,
        ncols=90,
    ) as bar:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            fh.write(chunk)
            bar.update(len(chunk))

    return "ok"


def build_manifest() -> list[tuple[str, Path]]:
    """Return (url, local_path) pairs for every EIA 930 balance file."""
    pairs = []
    for year in YEARS:
        for half in HALVES:
            name = f"EIA930_BALANCE_{year}_{half}.csv"
            pairs.append((f"{BASE_URL}/{name}", RAW_DIR / name))
    return pairs


def run_ingest(force: bool = False) -> dict[str, int]:
    """Download all files. Skip existing unless force=True."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest()
    counts = {"ok": 0, "skip": 0, "missing": 0, "error": 0}

    print(f"GridPulse Ingest — {len(manifest)} EIA 930 balance files (2019-2024)")
    print(f"Destination : {RAW_DIR.resolve()}")
    print(f"Force re-download: {force}\n")

    with requests.Session() as session, tqdm(
        manifest,
        desc="Overall",
        unit="file",
        ncols=90,
    ) as outer:
        for url, dest in outer:
            outer.set_postfix(file=dest.name)

            if dest.exists() and not force:
                if dest.stat().st_size >= _MIN_VALID_BYTES:
                    counts["skip"] += 1
                    outer.write(f"  [skip] {dest.name}")
                    continue
                outer.write(f"  [tiny] {dest.name} ({dest.stat().st_size / 1024:.0f} KB) — re-downloading")

            try:
                status = _download_one(url, dest, session)
            except requests.RequestException as exc:
                outer.write(f"  [err ] {dest.name}: {exc}")
                counts["error"] += 1
                continue

            if status == "missing":
                outer.write(f"  [404 ] {url}")
                counts["missing"] += 1
            else:
                size_mb = dest.stat().st_size / 1e6
                outer.write(f"  [done] {dest.name}  ({size_mb:.1f} MB)")
                counts["ok"] += 1

    print(
        f"\nFinished — downloaded={counts['ok']}  "
        f"skipped={counts['skip']}  "
        f"missing={counts['missing']}  "
        f"errors={counts['error']}"
    )
    return counts


def main():
    force = "--force" in sys.argv
    result = run_ingest(force=force)
    if result["error"] > 0:
        print("Some downloads failed — re-run to retry.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
