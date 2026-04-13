"""Run the affordability pipeline against a TrueLayer SQLite database.

Reads ``TRUELAYER_DB_PATH`` from the environment (defaults to
``./sandbox.db``) and optionally ``HOMEWARD_ACCOUNT_ID`` to pin a
specific account. Prints the resulting ``AffordabilityProfile`` as
pretty-printed JSON and writes it to
``<db-stem>_affordability_profile.json`` next to the database file.

Usage:

    python scripts/affordability_report.py
"""

from __future__ import annotations

import os
import sys

# Make ``homeward`` importable when running this script directly from
# the repo root without installing the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from homeward.affordability import build_profile


def main() -> int:
    db_path = os.environ.get("TRUELAYER_DB_PATH", "sandbox.db")
    account_id = os.environ.get("HOMEWARD_ACCOUNT_ID") or None
    window_months = int(os.environ.get("HOMEWARD_WINDOW_MONTHS", "6"))

    if not os.path.exists(db_path):
        print(
            f"error: database not found at {db_path!r}. "
            "Run homeward.truelayer.fetch_and_store_truelayer_data first "
            "or set TRUELAYER_DB_PATH.",
            file=sys.stderr,
        )
        return 1

    profile = build_profile(
        db_path=db_path,
        account_id=account_id,
        window_months=window_months,
    )

    json_blob = profile.to_json()
    print(json_blob)

    stem, _ = os.path.splitext(db_path)
    out_path = f"{stem}_affordability_profile.json"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json_blob)
    print(f"\nSaved to {out_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
