"""Fetch OpenBanking data and store it locally for analysis.

The storage layer uses SQLite with a normalized schema (accounts +
transactions) so analyses can be run with plain SQL or by loading the
tables into pandas via ``pd.read_sql``.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any, Iterable

import requests


_SCHEMA = """
CREATE TABLE IF NOT EXISTS accounts (
    account_id      TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL,
    nickname        TEXT,
    account_type    TEXT,
    account_subtype TEXT,
    currency        TEXT,
    fetched_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS transactions (
    transaction_id    TEXT PRIMARY KEY,
    account_id        TEXT NOT NULL REFERENCES accounts(account_id),
    booking_date      TEXT NOT NULL,
    value_date        TEXT,
    amount            REAL NOT NULL,
    currency          TEXT NOT NULL,
    credit_debit      TEXT NOT NULL,   -- 'Credit' or 'Debit'
    status            TEXT,
    merchant_name     TEXT,
    category          TEXT,
    description       TEXT,
    raw_json          TEXT NOT NULL    -- original payload for re-analysis
);

CREATE INDEX IF NOT EXISTS idx_txn_account_date
    ON transactions(account_id, booking_date);
CREATE INDEX IF NOT EXISTS idx_txn_category
    ON transactions(category);
"""


@dataclass(frozen=True)
class FetchResult:
    accounts_stored: int
    transactions_stored: int


def fetch_and_store_banking_data(
    access_token: str,
    user_id: str,
    db_path: str,
    api_base_url: str = "https://api.openbanking.example.com/open-banking/v3.1/aisp",
    timeout: float = 30.0,
) -> FetchResult:
    """Fetch a user's OpenBanking accounts + transactions and persist them.

    Idempotent: re-running upserts by primary key, so it's safe to schedule
    on a cron without producing duplicates.

    Args:
        access_token: OAuth2 bearer token obtained via the AISP consent flow.
        user_id: Your internal identifier for the user; stored alongside
            accounts so multi-user analyses stay scoped.
        db_path: Path to the SQLite file (created if missing).
        api_base_url: Root of the OpenBanking AISP API.
        timeout: Per-request HTTP timeout in seconds.

    Returns:
        A ``FetchResult`` with counts of rows written.
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

        accounts = _get_accounts(session, api_base_url, timeout)
        _upsert_accounts(conn, user_id, accounts)

        txn_count = 0
        for account in accounts:
            account_id = account["AccountId"]
            txns = _get_transactions(session, api_base_url, account_id, timeout)
            txn_count += _upsert_transactions(conn, account_id, txns)

        conn.commit()

    return FetchResult(
        accounts_stored=len(accounts),
        transactions_stored=txn_count,
    )


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


def _get_accounts(
    session: requests.Session, base_url: str, timeout: float
) -> list[dict[str, Any]]:
    resp = session.get(f"{base_url}/accounts", timeout=timeout)
    resp.raise_for_status()
    return resp.json().get("Data", {}).get("Account", [])


def _get_transactions(
    session: requests.Session,
    base_url: str,
    account_id: str,
    timeout: float,
) -> list[dict[str, Any]]:
    """Fetch all transactions for an account, following paginated ``Links.Next``."""
    results: list[dict[str, Any]] = []
    url: str | None = f"{base_url}/accounts/{account_id}/transactions"
    while url:
        resp = session.get(url, timeout=timeout)
        resp.raise_for_status()
        body = resp.json()
        results.extend(body.get("Data", {}).get("Transaction", []))
        url = body.get("Links", {}).get("Next")
    return results


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------


def _upsert_accounts(
    conn: sqlite3.Connection,
    user_id: str,
    accounts: Iterable[dict[str, Any]],
) -> None:
    rows = [
        (
            acct["AccountId"],
            user_id,
            acct.get("Nickname"),
            acct.get("AccountType"),
            acct.get("AccountSubType"),
            acct.get("Currency"),
            _now_iso(),
        )
        for acct in accounts
    ]
    conn.executemany(
        """
        INSERT INTO accounts (
            account_id, user_id, nickname, account_type,
            account_subtype, currency, fetched_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(account_id) DO UPDATE SET
            user_id = excluded.user_id,
            nickname = excluded.nickname,
            account_type = excluded.account_type,
            account_subtype = excluded.account_subtype,
            currency = excluded.currency,
            fetched_at = excluded.fetched_at
        """,
        rows,
    )


def _upsert_transactions(
    conn: sqlite3.Connection,
    account_id: str,
    transactions: Iterable[dict[str, Any]],
) -> int:
    import json

    rows = []
    for txn in transactions:
        amount_obj = txn.get("Amount", {})
        merchant = txn.get("MerchantDetails", {}) or {}
        category_code = txn.get("ProprietaryBankTransactionCode", {}) or {}
        rows.append(
            (
                txn["TransactionId"],
                account_id,
                txn.get("BookingDateTime"),
                txn.get("ValueDateTime"),
                float(amount_obj.get("Amount", 0.0)),
                amount_obj.get("Currency", ""),
                txn.get("CreditDebitIndicator", ""),
                txn.get("Status"),
                merchant.get("MerchantName"),
                category_code.get("Code"),
                txn.get("TransactionInformation"),
                json.dumps(txn, separators=(",", ":")),
            )
        )

    conn.executemany(
        """
        INSERT INTO transactions (
            transaction_id, account_id, booking_date, value_date,
            amount, currency, credit_debit, status,
            merchant_name, category, description, raw_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(transaction_id) DO UPDATE SET
            booking_date = excluded.booking_date,
            value_date = excluded.value_date,
            amount = excluded.amount,
            currency = excluded.currency,
            credit_debit = excluded.credit_debit,
            status = excluded.status,
            merchant_name = excluded.merchant_name,
            category = excluded.category,
            description = excluded.description,
            raw_json = excluded.raw_json
        """,
        rows,
    )
    return len(rows)


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
