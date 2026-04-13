"""Fetch a user's banking data from TrueLayer's Data API and store it locally.

This module is the ingestion half of the pipeline. It targets the TrueLayer
Data API v1 (``https://api.truelayer.com/data/v1``) and writes the results
into a local SQLite database. The analysis half lives in ``homeward.analysis``
and runs against the tables this module populates.

The schema is shaped around what TrueLayer actually returns and includes:

- ``accounts``: one row per linked bank account
- ``transactions``: full transaction history, with TrueLayer's category and
  hierarchical classification preserved
- ``standing_orders``: bank-registered standing orders (typically rent,
  regular transfers). Exact, not inferred.
- ``direct_debits``: bank-registered direct debits (utilities, subscriptions).
  Exact, not inferred.
- ``balances``: a timestamped balance snapshot per fetch, so cashflow
  forecasts can start from a known anchor.

All writes are upserts keyed on whatever identifier TrueLayer ships (or a
deterministic synthetic key for standing orders, which have none).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

import requests


TRUELAYER_API_BASE = "https://api.truelayer.com/data/v1"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS accounts (
    account_id      TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL,
    display_name    TEXT,
    account_type    TEXT,
    currency        TEXT,
    provider_id     TEXT,
    fetched_at      TEXT NOT NULL,
    raw_json        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS transactions (
    transaction_id              TEXT PRIMARY KEY,
    account_id                  TEXT NOT NULL REFERENCES accounts(account_id),
    timestamp                   TEXT NOT NULL,
    amount                      REAL NOT NULL,   -- signed: + credit, - debit
    currency                    TEXT NOT NULL,
    transaction_type            TEXT NOT NULL,   -- 'CREDIT' or 'DEBIT'
    transaction_category        TEXT,            -- TrueLayer enum
    transaction_classification  TEXT,            -- JSON-encoded hierarchy
    merchant_name               TEXT,
    description                 TEXT,
    running_balance             REAL,
    raw_json                    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_txn_account_ts
    ON transactions(account_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_txn_type
    ON transactions(transaction_type);
CREATE INDEX IF NOT EXISTS idx_txn_category
    ON transactions(transaction_category);

CREATE TABLE IF NOT EXISTS standing_orders (
    standing_order_key      TEXT PRIMARY KEY,    -- synthetic; see _standing_order_key()
    account_id              TEXT NOT NULL REFERENCES accounts(account_id),
    frequency               TEXT,
    status                  TEXT,
    reference               TEXT,
    first_payment_date      TEXT,
    first_payment_amount    REAL,
    next_payment_date       TEXT,
    next_payment_amount     REAL,
    final_payment_date      TEXT,
    final_payment_amount    REAL,
    currency                TEXT,
    fetched_at              TEXT NOT NULL,
    raw_json                TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS direct_debits (
    direct_debit_id             TEXT PRIMARY KEY,
    account_id                  TEXT NOT NULL REFERENCES accounts(account_id),
    name                        TEXT,
    status                      TEXT,
    previous_payment_timestamp  TEXT,
    previous_payment_amount     REAL,
    currency                    TEXT,
    fetched_at                  TEXT NOT NULL,
    raw_json                    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS balances (
    account_id          TEXT NOT NULL REFERENCES accounts(account_id),
    snapshot_at         TEXT NOT NULL,
    current_balance     REAL,
    available_balance   REAL,
    currency            TEXT,
    raw_json            TEXT NOT NULL,
    PRIMARY KEY (account_id, snapshot_at)
);
"""


@dataclass(frozen=True)
class FetchResult:
    accounts_stored: int
    transactions_stored: int
    standing_orders_stored: int
    direct_debits_stored: int
    balances_stored: int


def fetch_and_store_truelayer_data(
    access_token: str,
    user_id: str,
    db_path: str,
    api_base_url: str = TRUELAYER_API_BASE,
    timeout: float = 30.0,
) -> FetchResult:
    """Fetch accounts, transactions, standing orders, direct debits, and
    current balances from TrueLayer and persist them.

    Upserts are idempotent on primary keys, so re-running (e.g. after a
    client retry) won't duplicate rows.

    Args:
        access_token: TrueLayer OAuth2 bearer token for this user.
        user_id: Your internal identifier for the user.
        db_path: Path to the SQLite database file (created if missing).
        api_base_url: TrueLayer Data API base. Override for sandbox.
        timeout: Per-request HTTP timeout in seconds.
    """
    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
        }
    )

    with sqlite3.connect(db_path) as conn:
        conn.executescript(_SCHEMA)

        accounts = _get_results(session, f"{api_base_url}/accounts", timeout)
        _upsert_accounts(conn, user_id, accounts)

        now_iso = _now_iso()
        txn_count = so_count = dd_count = bal_count = 0

        for account in accounts:
            account_id = account["account_id"]

            txns = _get_results(
                session,
                f"{api_base_url}/accounts/{account_id}/transactions",
                timeout,
            )
            txn_count += _upsert_transactions(conn, account_id, txns)

            sos = _get_results(
                session,
                f"{api_base_url}/accounts/{account_id}/standing_orders",
                timeout,
            )
            so_count += _upsert_standing_orders(conn, account_id, sos, now_iso)

            dds = _get_results(
                session,
                f"{api_base_url}/accounts/{account_id}/direct_debits",
                timeout,
            )
            dd_count += _upsert_direct_debits(conn, account_id, dds, now_iso)

            bals = _get_results(
                session,
                f"{api_base_url}/accounts/{account_id}/balance",
                timeout,
            )
            bal_count += _upsert_balances(conn, account_id, bals, now_iso)

        conn.commit()

    return FetchResult(
        accounts_stored=len(accounts),
        transactions_stored=txn_count,
        standing_orders_stored=so_count,
        direct_debits_stored=dd_count,
        balances_stored=bal_count,
    )


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------


def _get_results(
    session: requests.Session, url: str, timeout: float
) -> list[dict[str, Any]]:
    """Call a TrueLayer Data API endpoint and return its ``results`` array.

    Empty results (e.g. an account with no standing orders) are returned as
    an empty list rather than raising.
    """
    resp = session.get(url, timeout=timeout)
    resp.raise_for_status()
    body = resp.json()
    return body.get("results", []) or []


# ---------------------------------------------------------------------------
# Upserts
# ---------------------------------------------------------------------------


def _upsert_accounts(
    conn: sqlite3.Connection,
    user_id: str,
    accounts: Iterable[dict[str, Any]],
) -> None:
    rows = [
        (
            acct["account_id"],
            user_id,
            acct.get("display_name"),
            acct.get("account_type"),
            acct.get("currency"),
            (acct.get("provider") or {}).get("provider_id"),
            _now_iso(),
            json.dumps(acct, separators=(",", ":")),
        )
        for acct in accounts
    ]
    conn.executemany(
        """
        INSERT INTO accounts (
            account_id, user_id, display_name, account_type,
            currency, provider_id, fetched_at, raw_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(account_id) DO UPDATE SET
            user_id = excluded.user_id,
            display_name = excluded.display_name,
            account_type = excluded.account_type,
            currency = excluded.currency,
            provider_id = excluded.provider_id,
            fetched_at = excluded.fetched_at,
            raw_json = excluded.raw_json
        """,
        rows,
    )


def _upsert_transactions(
    conn: sqlite3.Connection,
    account_id: str,
    transactions: Iterable[dict[str, Any]],
) -> int:
    rows = []
    for txn in transactions:
        txn_type = (txn.get("transaction_type") or "").upper()
        raw_amount = float(txn.get("amount") or 0.0)
        # Force sign: negative for debits, positive for credits. TrueLayer is
        # usually signed-correct already, but some providers return abs()
        # values and this keeps us consistent either way.
        if txn_type == "DEBIT" and raw_amount > 0:
            amount = -raw_amount
        elif txn_type == "CREDIT" and raw_amount < 0:
            amount = -raw_amount
        else:
            amount = raw_amount

        classification = txn.get("transaction_classification")
        classification_json = (
            json.dumps(classification) if classification is not None else None
        )

        running_balance = None
        rb = txn.get("running_balance")
        if isinstance(rb, dict) and rb.get("amount") is not None:
            running_balance = float(rb["amount"])

        rows.append(
            (
                txn["transaction_id"],
                account_id,
                txn.get("timestamp"),
                amount,
                txn.get("currency", ""),
                txn_type,
                txn.get("transaction_category"),
                classification_json,
                txn.get("merchant_name"),
                txn.get("description"),
                running_balance,
                json.dumps(txn, separators=(",", ":")),
            )
        )

    conn.executemany(
        """
        INSERT INTO transactions (
            transaction_id, account_id, timestamp, amount, currency,
            transaction_type, transaction_category, transaction_classification,
            merchant_name, description, running_balance, raw_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(transaction_id) DO UPDATE SET
            timestamp = excluded.timestamp,
            amount = excluded.amount,
            currency = excluded.currency,
            transaction_type = excluded.transaction_type,
            transaction_category = excluded.transaction_category,
            transaction_classification = excluded.transaction_classification,
            merchant_name = excluded.merchant_name,
            description = excluded.description,
            running_balance = excluded.running_balance,
            raw_json = excluded.raw_json
        """,
        rows,
    )
    return len(rows)


def _upsert_standing_orders(
    conn: sqlite3.Connection,
    account_id: str,
    standing_orders: Iterable[dict[str, Any]],
    fetched_at: str,
) -> int:
    rows = []
    for so in standing_orders:
        rows.append(
            (
                _standing_order_key(account_id, so),
                account_id,
                so.get("frequency"),
                so.get("status"),
                so.get("reference"),
                so.get("first_payment_date"),
                _amount_field(so.get("first_payment_amount")),
                so.get("next_payment_date"),
                _amount_field(so.get("next_payment_amount")),
                so.get("final_payment_date"),
                _amount_field(so.get("final_payment_amount")),
                so.get("currency"),
                fetched_at,
                json.dumps(so, separators=(",", ":")),
            )
        )

    conn.executemany(
        """
        INSERT INTO standing_orders (
            standing_order_key, account_id, frequency, status, reference,
            first_payment_date, first_payment_amount,
            next_payment_date, next_payment_amount,
            final_payment_date, final_payment_amount,
            currency, fetched_at, raw_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(standing_order_key) DO UPDATE SET
            frequency = excluded.frequency,
            status = excluded.status,
            reference = excluded.reference,
            first_payment_date = excluded.first_payment_date,
            first_payment_amount = excluded.first_payment_amount,
            next_payment_date = excluded.next_payment_date,
            next_payment_amount = excluded.next_payment_amount,
            final_payment_date = excluded.final_payment_date,
            final_payment_amount = excluded.final_payment_amount,
            currency = excluded.currency,
            fetched_at = excluded.fetched_at,
            raw_json = excluded.raw_json
        """,
        rows,
    )
    return len(rows)


def _upsert_direct_debits(
    conn: sqlite3.Connection,
    account_id: str,
    direct_debits: Iterable[dict[str, Any]],
    fetched_at: str,
) -> int:
    rows = []
    for dd in direct_debits:
        rows.append(
            (
                dd["direct_debit_id"],
                account_id,
                dd.get("name"),
                dd.get("status"),
                dd.get("previous_payment_timestamp"),
                _amount_field(dd.get("previous_payment_amount")),
                dd.get("currency"),
                fetched_at,
                json.dumps(dd, separators=(",", ":")),
            )
        )

    conn.executemany(
        """
        INSERT INTO direct_debits (
            direct_debit_id, account_id, name, status,
            previous_payment_timestamp, previous_payment_amount,
            currency, fetched_at, raw_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(direct_debit_id) DO UPDATE SET
            name = excluded.name,
            status = excluded.status,
            previous_payment_timestamp = excluded.previous_payment_timestamp,
            previous_payment_amount = excluded.previous_payment_amount,
            currency = excluded.currency,
            fetched_at = excluded.fetched_at,
            raw_json = excluded.raw_json
        """,
        rows,
    )
    return len(rows)


def _upsert_balances(
    conn: sqlite3.Connection,
    account_id: str,
    balances: Iterable[dict[str, Any]],
    snapshot_at: str,
) -> int:
    # The balance endpoint returns a single-element list by convention, but
    # we loop for safety.
    rows = []
    for bal in balances:
        rows.append(
            (
                account_id,
                snapshot_at,
                _as_float(bal.get("current")),
                _as_float(bal.get("available")),
                bal.get("currency"),
                json.dumps(bal, separators=(",", ":")),
            )
        )

    conn.executemany(
        """
        INSERT INTO balances (
            account_id, snapshot_at, current_balance,
            available_balance, currency, raw_json
        ) VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(account_id, snapshot_at) DO UPDATE SET
            current_balance = excluded.current_balance,
            available_balance = excluded.available_balance,
            currency = excluded.currency,
            raw_json = excluded.raw_json
        """,
        rows,
    )
    return len(rows)


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def _standing_order_key(account_id: str, so: dict[str, Any]) -> str:
    """TrueLayer doesn't give standing orders a stable ID. We synthesize one
    from fields that together uniquely identify a standing order on a given
    account: reference, first_payment_date, and first_payment_amount. If a
    standing order is cancelled and re-created identically, we'll treat it
    as the same row — acceptable for analysis.
    """
    parts = (
        account_id,
        so.get("reference") or "",
        so.get("first_payment_date") or "",
        str(_amount_field(so.get("first_payment_amount")) or ""),
    )
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()


def _amount_field(value: Any) -> float | None:
    """TrueLayer sometimes returns amounts as scalars, sometimes as
    ``{amount, currency}`` objects. Normalise to a float.
    """
    if value is None:
        return None
    if isinstance(value, dict):
        inner = value.get("amount")
        return float(inner) if inner is not None else None
    return float(value)


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
