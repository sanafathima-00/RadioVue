"""Command line interface for the standalone standardization MVP."""

from __future__ import annotations

import argparse
import json

from .pipeline import StandardizationError, standardize
from .profiles import profile_names


def main() -> int:
    parser = argparse.ArgumentParser(description="Conservative PA X-ray standardization for SV-DRR")
    parser.add_argument("--input", required=True, help="DICOM, PNG, or JPEG source; never modified")
    parser.add_argument("--output-dir", required=True, help="New directory for derived outputs")
    parser.add_argument("--profile", choices=profile_names(), required=True)
    parser.add_argument("--projection", choices=["pa", "ap", "lateral", "oblique"], help="Explicit assertion only when metadata is missing")
    args = parser.parse_args()
    try:
        result = standardize(args.input, args.output_dir, args.profile, args.projection)
    except StandardizationError as error:
        parser.error(str(error))
    print(json.dumps({key: str(value) for key, value in result.items() if key != "report"}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
