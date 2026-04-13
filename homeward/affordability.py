"""Homeward affordability pipeline.

Produces a structured, auditable affordability profile from a TrueLayer
SQLite database (populated by ``homeward.truelayer``). Intended for
FCA-compliant mortgage decisioning: every number is derived by
deterministic Python code with explainable rules, not a model.

Downstream consumers:

- **Underwriters** read the :class:`AffordabilityProfile` dataclass
  directly (or its JSON form).
- **An optional LLM narration layer** receives only
  ``AffordabilityProfile.to_json()`` — the already-computed numbers —
  and generates a human-readable summary. The LLM never sees raw
  transactions, which keeps the audit trail clean.

Pipeline stages (one private function per stage):

1. Pick the primary transaction account.
2. Detect income sources — cluster credit transactions by normalised
   sender and pick streams that recur monthly with stable amounts.
3. Compute committed expenditure from the bank-sourced standing orders
   and direct debits (exact, not inferred).
4. Summarise discretionary spend by TrueLayer's transaction
   classification.
5. Flag stress indicators — gambling, payday loans, returned payments,
   overdraft, low balance.
6. Compute net disposable income and debt-to-income ratio.
7. Summarise month-over-month net flow as a trajectory.

All of this is assembled in :func:`build_profile`, which returns an
:class:`AffordabilityProfile` dataclass.
"""

from __future__ import annotations

import json
import sqlite3
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from homeward.analysis import (
    _NON_DISCRETIONARY,
    _classify_outgoing,
    _frequency_from_truelayer_code,
    _monthly_equivalent,
    _normalise_label,
    _top_level_classification,
    _utc_now,
)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class IncomeSource:
    label: str              # normalised sender label
    avg_monthly: float
    months_seen: int
    variance_pct: float     # coefficient of variation, 0.0 - 1.0+
    confidence: str         # 'HIGH' | 'MEDIUM' | 'LOW'


@dataclass
class CommittedOutgoing:
    source: str             # 'standing_order' | 'direct_debit'
    label: str
    classification: str     # 'rent' | 'utility' | 'subscription' | ...
    monthly_amount: float


@dataclass
class DiscretionaryCategory:
    category: str           # top-level TrueLayer classification
    total_spent: float
    txn_count: int
    monthly_average: float


@dataclass
class StressFlag:
    flag: str               # 'GAMBLING' | 'PAYDAY_LOAN' | 'RETURNED_PAYMENT' | 'OVERDRAFT' | 'LOW_BALANCE'
    severity: str           # 'HIGH' | 'MEDIUM' | 'LOW'
    count: int
    detail: str | None = None


@dataclass
class BalanceMonth:
    month: str              # 'YYYY-MM'
    net_flow: float
    txn_count: int


@dataclass
class AffordabilityProfile:
    generated_at: str
    window_months: int
    accounts: list[dict[str, Any]]
    primary_account_id: str | None
    income_sources: list[IncomeSource]
    total_monthly_income: float
    committed_outgoings: list[CommittedOutgoing]
    total_monthly_committed: float
    discretionary_by_category: list[DiscretionaryCategory]
    total_monthly_discretionary: float
    stress_flags: list[StressFlag]
    net_disposable_income: float
    debt_to_income_ratio: float | None
    balance_trajectory: list[BalanceMonth]
    max_mortgage_monthly_guideline: float   # 0.4 * NDI — prototype rule only
    income_confidence: str                  # summary field: highest confidence among income sources

    def to_json(self) -> str:
        """Serialise to a pretty-printed JSON string.

        This is the exact string you hand to the LLM narration layer.
        It contains only the computed numbers, never raw transactions,
        so the LLM can't hallucinate about individual spending.
        """
        return json.dumps(asdict(self), indent=2, default=str)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_profile(
    db_path: str,
    account_id: str | None = None,
    window_months: int = 6,
) -> AffordabilityProfile:
    """Build an :class:`AffordabilityProfile` from a TrueLayer SQLite DB.

    Args:
        db_path: Path to the SQLite file populated by
            ``homeward.truelayer.fetch_and_store_truelayer_data``.
        account_id: Specific account to analyse. If ``None`` (default),
            the first account with ``account_type = 'TRANSACTION'`` is
            picked. Joint accounts and savings accounts should be
            analysed explicitly by passing the id.
        window_months: Affordability window in months. Defaults to 6,
            which is the common FCA standard for income assessment.

    Returns:
        A populated :class:`AffordabilityProfile`. Every field is
        always set; empty lists indicate "nothing detected" rather than
        "not attempted".
    """
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row

        accounts = [
            dict(r)
            for r in conn.execute(
                "SELECT account_id, display_name, account_type, currency, provider_id FROM accounts"
            )
        ]
        if account_id is None:
            account_id = _pick_primary_account(conn)

        income_sources = (
            _detect_income_sources(conn, account_id, window_months)
            if account_id
            else []
        )
        committed = (
            _committed_expenditure(conn, account_id) if account_id else []
        )
        discretionary = (
            _discretionary_spend(conn, account_id, window_months)
            if account_id
            else []
        )
        stress_flags = _stress_flags(conn, account_id, window_months)
        trajectory = (
            _balance_trajectory(conn, account_id, window_months)
            if account_id
            else []
        )

    total_monthly_income = round(sum(s.avg_monthly for s in income_sources), 2)
    total_monthly_committed = round(
        sum(o.monthly_amount for o in committed), 2
    )
    total_monthly_discretionary = round(
        sum(c.monthly_average for c in discretionary), 2
    )

    ndi = round(
        total_monthly_income - total_monthly_committed - total_monthly_discretionary,
        2,
    )
    dti = (
        round(total_monthly_committed / total_monthly_income, 3)
        if total_monthly_income
        else None
    )
    max_mortgage = round(max(ndi, 0.0) * 0.4, 2)

    income_confidence = _summary_confidence(income_sources)

    return AffordabilityProfile(
        generated_at=_utc_now().isoformat(),
        window_months=window_months,
        accounts=accounts,
        primary_account_id=account_id,
        income_sources=income_sources,
        total_monthly_income=total_monthly_income,
        committed_outgoings=committed,
        total_monthly_committed=total_monthly_committed,
        discretionary_by_category=discretionary,
        total_monthly_discretionary=total_monthly_discretionary,
        stress_flags=stress_flags,
        net_disposable_income=ndi,
        debt_to_income_ratio=dti,
        balance_trajectory=trajectory,
        max_mortgage_monthly_guideline=max_mortgage,
        income_confidence=income_confidence,
    )


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------


def _pick_primary_account(conn: sqlite3.Connection) -> str | None:
    row = conn.execute(
        """
        SELECT account_id
        FROM accounts
        WHERE upper(account_type) = 'TRANSACTION'
        ORDER BY account_id
        LIMIT 1
        """
    ).fetchone()
    if row:
        return row["account_id"]
    # Fallback: any account at all
    any_row = conn.execute(
        "SELECT account_id FROM accounts ORDER BY account_id LIMIT 1"
    ).fetchone()
    return any_row["account_id"] if any_row else None


def _detect_income_sources(
    conn: sqlite3.Connection,
    account_id: str,
    window_months: int,
) -> list[IncomeSource]:
    """Identify likely income streams.

    Rule (explainable in one sentence for an underwriter): group credit
    transactions by **normalised sender** — that is, the description
    with reference numbers and dates stripped — and keep groups that
    appear in **at least 3 distinct months** with a **coefficient of
    variation below 10%**. Confidence is ``HIGH`` if variation is below
    2%, ``MEDIUM`` below 10%, otherwise ``LOW``.
    """
    cutoff_iso = _months_ago_iso(window_months)

    # Bucket credits by normalised sender, tracking the monthly total
    # per bucket (so multiple same-month credits like overtime top-ups
    # get rolled into one monthly data point).
    buckets: dict[str, dict[str, float]] = defaultdict(dict)

    rows = conn.execute(
        """
        SELECT timestamp, amount, description
        FROM transactions
        WHERE account_id = ?
          AND transaction_type = 'CREDIT'
          AND timestamp >= ?
        """,
        (account_id, cutoff_iso),
    ).fetchall()

    for row in rows:
        label = _normalise_label(row["description"] or "")
        if not label:
            continue
        month = row["timestamp"][:7]   # 'YYYY-MM'
        amount = float(row["amount"])
        if amount <= 0:
            continue
        buckets[label][month] = buckets[label].get(month, 0.0) + amount

    sources: list[IncomeSource] = []
    for label, by_month in buckets.items():
        if len(by_month) < 3:
            continue
        amounts = list(by_month.values())
        mean = statistics.mean(amounts)
        if mean <= 0:
            continue
        stdev = statistics.pstdev(amounts) if len(amounts) > 1 else 0.0
        cv = stdev / mean if mean else 1.0
        if cv > 0.10:
            continue
        if cv < 0.02:
            confidence = "HIGH"
        elif cv < 0.10:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
        sources.append(
            IncomeSource(
                label=label,
                avg_monthly=round(mean, 2),
                months_seen=len(by_month),
                variance_pct=round(cv, 4),
                confidence=confidence,
            )
        )

    sources.sort(key=lambda s: s.avg_monthly, reverse=True)
    return sources


def _committed_expenditure(
    conn: sqlite3.Connection,
    account_id: str,
) -> list[CommittedOutgoing]:
    """Read bank-sourced standing orders and direct debits.

    These are exact, not inferred — the bank's own registered recurring
    payments. Weekly/fortnightly standing orders are converted to their
    monthly-equivalent amount so the NDI calculation is consistent.
    """
    results: list[CommittedOutgoing] = []

    for so in conn.execute(
        """
        SELECT reference, frequency, next_payment_amount, first_payment_amount, status
        FROM standing_orders
        WHERE account_id = ?
          AND (status IS NULL OR upper(status) != 'CANCELLED')
        """,
        (account_id,),
    ):
        amount = so["next_payment_amount"] or so["first_payment_amount"] or 0.0
        if not amount:
            continue
        frequency = _frequency_from_truelayer_code(so["frequency"]) or "MONTHLY"
        monthly = _monthly_equivalent(abs(float(amount)), frequency)
        label = (so["reference"] or "").strip() or "standing order"
        results.append(
            CommittedOutgoing(
                source="standing_order",
                label=label,
                classification=_classify_outgoing(label),
                monthly_amount=round(monthly, 2),
            )
        )

    for dd in conn.execute(
        """
        SELECT name, previous_payment_amount, status
        FROM direct_debits
        WHERE account_id = ?
          AND (status IS NULL OR upper(status) != 'CANCELLED')
        """,
        (account_id,),
    ):
        amount = dd["previous_payment_amount"] or 0.0
        if not amount:
            continue
        # Direct debits don't expose a frequency; most are monthly.
        # Inferring frequency from history is a future enhancement.
        monthly = abs(float(amount))
        label = (dd["name"] or "").strip() or "direct debit"
        results.append(
            CommittedOutgoing(
                source="direct_debit",
                label=label,
                classification=_classify_outgoing(label),
                monthly_amount=round(monthly, 2),
            )
        )

    results.sort(key=lambda o: o.monthly_amount, reverse=True)
    return results


def _discretionary_spend(
    conn: sqlite3.Connection,
    account_id: str,
    window_months: int,
) -> list[DiscretionaryCategory]:
    """Summarise debit spending by TrueLayer's top-level classification.

    Non-discretionary categories (rent, bills, groceries, transport,
    insurance, healthcare) are excluded — they're tracked via the
    committed-expenditure section, and the NDI calc subtracts them
    separately. Remaining spend is the user's discretionary bucket.
    """
    cutoff_iso = _months_ago_iso(window_months)

    rows = conn.execute(
        """
        SELECT amount, transaction_classification
        FROM transactions
        WHERE account_id = ?
          AND transaction_type = 'DEBIT'
          AND timestamp >= ?
        """,
        (account_id, cutoff_iso),
    ).fetchall()

    per_cat: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        category = _top_level_classification(row["transaction_classification"])
        if category in _NON_DISCRETIONARY:
            continue
        per_cat[category].append(abs(float(row["amount"])))

    window_months = max(window_months, 1)
    results = [
        DiscretionaryCategory(
            category=category,
            total_spent=round(sum(amounts), 2),
            txn_count=len(amounts),
            monthly_average=round(sum(amounts) / window_months, 2),
        )
        for category, amounts in per_cat.items()
    ]
    results.sort(key=lambda c: c.monthly_average, reverse=True)
    return results


def _stress_flags(
    conn: sqlite3.Connection,
    account_id: str | None,
    window_months: int,
) -> list[StressFlag]:
    """Surface financial-stress signals relevant to affordability.

    Rules are intentionally conservative and keyword-based so each
    flag is traceable to a specific SQL filter an auditor can cite.
    """
    flags: list[StressFlag] = []
    if account_id is None:
        return flags

    cutoff_iso = _months_ago_iso(window_months)

    def count_where(extra_sql: str, params: tuple[Any, ...]) -> int:
        row = conn.execute(
            f"""
            SELECT COUNT(*) FROM transactions
            WHERE account_id = ? AND timestamp >= ? {extra_sql}
            """,
            (account_id, cutoff_iso, *params),
        ).fetchone()
        return int(row[0]) if row else 0

    # Gambling. Classification is stored as a JSON array string
    # (e.g. '["Entertainment","Gambling"]'), so a LIKE on the quoted
    # element name is more reliable than a bare substring match.
    gambling = count_where(
        """
        AND (
            lower(transaction_classification) LIKE ?
            OR upper(description) LIKE ?
            OR upper(description) LIKE ?
            OR upper(description) LIKE ?
            OR upper(description) LIKE ?
            OR upper(description) LIKE ?
        )
        """,
        ('%"gambling"%', "%BET365%", "%PADDY POWER%", "%BETFAIR%", "%WILLIAM HILL%", "%LADBROKES%"),
    )
    if gambling > 0:
        flags.append(
            StressFlag(
                flag="GAMBLING",
                severity="HIGH",
                count=gambling,
                detail=f"{gambling} gambling transactions in last {window_months} months",
            )
        )

    # Payday loans
    payday = count_where(
        """
        AND (
            upper(description) LIKE ?
            OR upper(description) LIKE ?
            OR upper(description) LIKE ?
            OR upper(description) LIKE ?
        )
        """,
        ("%WONGA%", "%QUICKQUID%", "%PAYDAY%", "%SUNNY LOANS%"),
    )
    if payday > 0:
        flags.append(
            StressFlag(
                flag="PAYDAY_LOAN",
                severity="HIGH",
                count=payday,
                detail=f"{payday} payday-loan-pattern transactions",
            )
        )

    # Returned / unpaid payments
    returned = count_where(
        """
        AND (
            upper(description) LIKE ?
            OR upper(description) LIKE ?
            OR upper(description) LIKE ?
        )
        """,
        ("%RETURNED%", "%UNPAID%", "%REFER TO DRAWER%"),
    )
    if returned > 0:
        flags.append(
            StressFlag(
                flag="RETURNED_PAYMENT",
                severity="MEDIUM",
                count=returned,
                detail=f"{returned} returned or unpaid transactions",
            )
        )

    # Balance stress: check the latest snapshot for this account.
    bal = conn.execute(
        """
        SELECT current_balance
        FROM balances
        WHERE account_id = ?
        ORDER BY snapshot_at DESC
        LIMIT 1
        """,
        (account_id,),
    ).fetchone()
    if bal and bal["current_balance"] is not None:
        cb = float(bal["current_balance"])
        if cb < 0:
            flags.append(
                StressFlag(
                    flag="OVERDRAFT",
                    severity="HIGH",
                    count=1,
                    detail=f"Latest balance is {cb:.2f} (overdrawn)",
                )
            )
        elif cb < 50:
            flags.append(
                StressFlag(
                    flag="LOW_BALANCE",
                    severity="MEDIUM",
                    count=1,
                    detail=f"Latest balance is {cb:.2f} (below £50)",
                )
            )

    return flags


def _balance_trajectory(
    conn: sqlite3.Connection,
    account_id: str,
    window_months: int,
) -> list[BalanceMonth]:
    """Month-over-month net flow for the account.

    This is **net flow**, not a running balance — we don't have enough
    historical balance snapshots to reconstruct the latter. A negative
    ``net_flow`` in a given month means the account spent more than it
    received that month; a positive number means it accumulated.
    """
    cutoff_iso = _months_ago_iso(window_months)

    rows = conn.execute(
        """
        SELECT strftime('%Y-%m', timestamp) AS month,
               ROUND(SUM(amount), 2) AS net_flow,
               COUNT(*) AS txn_count
        FROM transactions
        WHERE account_id = ? AND timestamp >= ?
        GROUP BY month
        ORDER BY month
        """,
        (account_id, cutoff_iso),
    ).fetchall()

    return [
        BalanceMonth(
            month=r["month"],
            net_flow=float(r["net_flow"] or 0.0),
            txn_count=int(r["txn_count"]),
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def _summary_confidence(sources: list[IncomeSource]) -> str:
    if not sources:
        return "UNDETECTED"
    ranking = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
    best = max(sources, key=lambda s: ranking.get(s.confidence, 0))
    return best.confidence


def _months_ago_iso(months: int) -> str:
    """ISO timestamp for ``months`` ago (30-day approximation)."""
    from datetime import timedelta

    return (datetime.now(timezone.utc) - timedelta(days=months * 30)).isoformat()
