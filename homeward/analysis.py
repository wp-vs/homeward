"""Analyse stored TrueLayer banking data.

Runs against the SQLite database populated by ``homeward.truelayer`` and
produces four kinds of output:

1. ``detect_wages`` — estimates the user's salary by clustering recurring
   credits (TrueLayer does not auto-classify credit transactions, so this
   part is detection rather than lookup).
2. ``detect_significant_outgoings`` — enumerates the user's recurring
   outflows directly from the ``standing_orders`` and ``direct_debits``
   tables, tagging each with a rough classification (rent / utility /
   subscription / insurance / savings / other).
3. ``spending_profile`` — summarises spending over a rolling window, split
   by TrueLayer's transaction classification and into discretionary vs
   non-discretionary buckets.
4. ``forecast_cashflow`` — projects the account balance forward N days by
   replaying expected recurring events plus an average daily discretionary
   burn rate.

These are estimates, not ground truth. Each dataclass carries a
``confidence`` or documents its assumptions so the caller knows how much
to trust it.
"""

from __future__ import annotations

import json
import re
import sqlite3
import statistics
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WageEstimate:
    source_label: str           # normalised sender / merchant name
    avg_amount: float           # median of the matching credits
    frequency: str              # 'WEEKLY' | 'FORTNIGHTLY' | 'MONTHLY' | ...
    last_paid: date
    next_expected: date
    confidence: float           # 0.0 - 1.0
    occurrences: int            # how many transactions the estimate was built from


@dataclass(frozen=True)
class OutgoingEstimate:
    source: str                 # 'standing_order' or 'direct_debit'
    label: str                  # reference or creditor name
    classification: str         # 'rent' | 'utility' | 'subscription' | ...
    amount: float               # positive number, most recent payment
    frequency: str              # normalised frequency string
    monthly_equivalent: float   # amount normalised to a monthly cadence
    next_expected: date | None


@dataclass(frozen=True)
class CategorySummary:
    category: str
    total_spent: float
    txn_count: int
    mean_amount: float


@dataclass(frozen=True)
class SpendingProfile:
    window_start: date
    window_end: date
    total_income: float
    total_spending: float
    net: float
    by_category: list[CategorySummary]
    discretionary_total: float
    non_discretionary_total: float


@dataclass(frozen=True)
class CashflowEvent:
    on_date: date
    amount: float               # signed: positive inflow, negative outflow
    label: str
    kind: str                   # 'wage' | 'outgoing' | 'discretionary'


@dataclass(frozen=True)
class CashflowForecast:
    start_date: date
    start_balance: float
    daily_projection: list[tuple[date, float]]   # (date, projected_balance)
    events: list[CashflowEvent]
    assumptions: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Public API: wages
# ---------------------------------------------------------------------------


def detect_wages(
    db_path: str,
    account_id: str,
    lookback_days: int = 180,
) -> WageEstimate | None:
    """Estimate the user's wages for a given account.

    Clusters credit transactions by a normalised sender label, finds the
    clusters whose timing looks periodic, and returns the one with the
    highest median amount weighted by regularity. Returns ``None`` if no
    cluster meets the minimum evidence threshold (3 occurrences).
    """
    cutoff_iso = _iso_days_ago(lookback_days)

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT timestamp, amount, description, merchant_name
            FROM transactions
            WHERE account_id = ?
              AND transaction_type = 'CREDIT'
              AND timestamp >= ?
            ORDER BY timestamp
            """,
            (account_id, cutoff_iso),
        ).fetchall()

    clusters: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        label = row["merchant_name"] or _normalise_label(row["description"] or "")
        if not label:
            continue
        clusters.setdefault(label, []).append(
            {
                "ts": _parse_iso(row["timestamp"]),
                "amount": float(row["amount"]),
            }
        )

    best_score = 0.0
    best: WageEstimate | None = None

    for label, txns in clusters.items():
        if len(txns) < 3:
            continue
        txns.sort(key=lambda t: t["ts"])
        gaps_days = [
            (txns[i + 1]["ts"] - txns[i]["ts"]).total_seconds() / 86400.0
            for i in range(len(txns) - 1)
        ]
        frequency, freq_conf = _classify_frequency(gaps_days)
        if frequency is None:
            continue

        amounts = [t["amount"] for t in txns]
        median_amount = statistics.median(amounts)
        if median_amount <= 0:
            continue

        # Coefficient of variation on amounts: 0 means rock-steady pay;
        # > ~0.3 means highly variable (probably not salary).
        cv = (
            statistics.pstdev(amounts) / median_amount
            if median_amount
            else 1.0
        )
        amount_conf = max(0.0, 1.0 - cv)
        confidence = min(1.0, freq_conf * (0.5 + 0.5 * amount_conf))

        # Weight score by median amount — salary is typically the largest
        # regular credit on the account.
        score = median_amount * confidence

        if score > best_score:
            last_paid = txns[-1]["ts"]
            period_days = _FREQUENCY_DAYS[frequency]
            best_score = score
            best = WageEstimate(
                source_label=label,
                avg_amount=round(median_amount, 2),
                frequency=frequency,
                last_paid=last_paid.date(),
                next_expected=(last_paid + timedelta(days=period_days)).date(),
                confidence=round(confidence, 3),
                occurrences=len(txns),
            )

    return best


# ---------------------------------------------------------------------------
# Public API: significant outgoings
# ---------------------------------------------------------------------------


def detect_significant_outgoings(
    db_path: str,
    account_id: str,
    min_monthly_amount: float = 10.0,
) -> list[OutgoingEstimate]:
    """Enumerate the user's significant recurring outgoings.

    Reads directly from the ``standing_orders`` and ``direct_debits``
    tables — these are bank-sourced, so no detection is required. For
    each entry we classify it (rent / utility / subscription / ...) and
    compute its monthly-equivalent amount. The result is filtered to
    items above ``min_monthly_amount`` and sorted by that value.
    """
    results: list[OutgoingEstimate] = []

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row

        for so in conn.execute(
            """
            SELECT reference, frequency, next_payment_date, next_payment_amount,
                   first_payment_amount, status
            FROM standing_orders
            WHERE account_id = ? AND (status IS NULL OR status != 'CANCELLED')
            """,
            (account_id,),
        ):
            amount = so["next_payment_amount"] or so["first_payment_amount"] or 0.0
            if not amount:
                continue
            amount = abs(float(amount))
            frequency = _frequency_from_truelayer_code(so["frequency"]) or "MONTHLY"
            monthly = _monthly_equivalent(amount, frequency)
            if monthly < min_monthly_amount:
                continue
            label = (so["reference"] or "").strip() or "standing order"
            results.append(
                OutgoingEstimate(
                    source="standing_order",
                    label=label,
                    classification=_classify_outgoing(label),
                    amount=round(amount, 2),
                    frequency=frequency,
                    monthly_equivalent=round(monthly, 2),
                    next_expected=_parse_date(so["next_payment_date"]),
                )
            )

        for dd in conn.execute(
            """
            SELECT name, previous_payment_timestamp, previous_payment_amount, status
            FROM direct_debits
            WHERE account_id = ? AND (status IS NULL OR status != 'CANCELLED')
            """,
            (account_id,),
        ):
            amount = dd["previous_payment_amount"] or 0.0
            if not amount:
                continue
            amount = abs(float(amount))
            # Direct debits don't expose a frequency field. Most are
            # monthly in practice; a proper version would infer from the
            # transaction history. MONTHLY is a reasonable default.
            frequency = "MONTHLY"
            monthly = _monthly_equivalent(amount, frequency)
            if monthly < min_monthly_amount:
                continue
            label = (dd["name"] or "").strip() or "direct debit"
            prev_ts = _parse_iso_date(dd["previous_payment_timestamp"])
            next_expected = (
                prev_ts + timedelta(days=30) if prev_ts is not None else None
            )
            results.append(
                OutgoingEstimate(
                    source="direct_debit",
                    label=label,
                    classification=_classify_outgoing(label),
                    amount=round(amount, 2),
                    frequency=frequency,
                    monthly_equivalent=round(monthly, 2),
                    next_expected=next_expected,
                )
            )

    results.sort(key=lambda o: o.monthly_equivalent, reverse=True)
    return results


# ---------------------------------------------------------------------------
# Public API: spending profile
# ---------------------------------------------------------------------------


def spending_profile(
    db_path: str,
    account_id: str,
    window_months: int = 3,
) -> SpendingProfile:
    """Summarise inflows and outflows over a rolling window.

    Groups spending by the top level of TrueLayer's
    ``transaction_classification`` (e.g. ``"Bills and Utilities"``,
    ``"Food & Dining"``). Transactions without a classification fall into
    an ``"Uncategorised"`` bucket.
    """
    window_days = int(window_months * 30)
    cutoff = _utc_now() - timedelta(days=window_days)
    cutoff_iso = cutoff.isoformat()

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT amount, transaction_type, transaction_classification
            FROM transactions
            WHERE account_id = ? AND timestamp >= ?
            """,
            (account_id, cutoff_iso),
        ).fetchall()

    total_income = 0.0
    total_spending = 0.0
    cat_totals: dict[str, list[float]] = {}

    for row in rows:
        amount = float(row["amount"])
        if row["transaction_type"] == "CREDIT":
            total_income += amount
            continue
        # Debit: treat as a positive spending number for reporting.
        spend = abs(amount)
        total_spending += spend
        category = _top_level_classification(row["transaction_classification"])
        cat_totals.setdefault(category, []).append(spend)

    by_category = [
        CategorySummary(
            category=cat,
            total_spent=round(sum(spends), 2),
            txn_count=len(spends),
            mean_amount=round(statistics.mean(spends), 2),
        )
        for cat, spends in cat_totals.items()
    ]
    by_category.sort(key=lambda c: c.total_spent, reverse=True)

    discretionary = sum(
        c.total_spent for c in by_category if c.category not in _NON_DISCRETIONARY
    )
    non_discretionary = sum(
        c.total_spent for c in by_category if c.category in _NON_DISCRETIONARY
    )

    return SpendingProfile(
        window_start=cutoff.date(),
        window_end=_utc_now().date(),
        total_income=round(total_income, 2),
        total_spending=round(total_spending, 2),
        net=round(total_income - total_spending, 2),
        by_category=by_category,
        discretionary_total=round(discretionary, 2),
        non_discretionary_total=round(non_discretionary, 2),
    )


# ---------------------------------------------------------------------------
# Public API: cashflow forecast
# ---------------------------------------------------------------------------


def forecast_cashflow(
    db_path: str,
    account_id: str,
    days: int = 60,
) -> CashflowForecast:
    """Project the account balance forward ``days`` days.

    Starts from the most recent balance snapshot in the ``balances`` table
    (anchored by ``homeward.truelayer``) and applies:

    - Expected wage inflows on predicted dates (from ``detect_wages``)
    - Expected outgoings on predicted dates (from
      ``detect_significant_outgoings``)
    - A flat daily discretionary burn rate computed from the spending
      profile's discretionary total

    This is a projection, not a prediction. It assumes the recent pattern
    continues unchanged.
    """
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        bal_row = conn.execute(
            """
            SELECT snapshot_at, current_balance
            FROM balances
            WHERE account_id = ?
            ORDER BY snapshot_at DESC
            LIMIT 1
            """,
            (account_id,),
        ).fetchone()

    if bal_row is None or bal_row["current_balance"] is None:
        raise ValueError(
            f"No balance snapshot stored for account {account_id}; "
            "run fetch_and_store_truelayer_data first."
        )

    start_date = _utc_now().date()
    start_balance = float(bal_row["current_balance"])
    end_date = start_date + timedelta(days=days)

    events: list[CashflowEvent] = []

    # 1. Wage inflows
    wage = detect_wages(db_path, account_id)
    if wage is not None:
        period = _FREQUENCY_DAYS[wage.frequency]
        cursor = wage.next_expected
        while cursor <= end_date:
            if cursor >= start_date:
                events.append(
                    CashflowEvent(
                        on_date=cursor,
                        amount=wage.avg_amount,
                        label=wage.source_label,
                        kind="wage",
                    )
                )
            cursor = cursor + timedelta(days=period)

    # 2. Recurring outflows
    outgoings = detect_significant_outgoings(db_path, account_id)
    for out in outgoings:
        period = _FREQUENCY_DAYS.get(out.frequency, 30)
        cursor = out.next_expected or (start_date + timedelta(days=period))
        while cursor <= end_date:
            if cursor >= start_date:
                events.append(
                    CashflowEvent(
                        on_date=cursor,
                        amount=-out.amount,
                        label=out.label,
                        kind="outgoing",
                    )
                )
            cursor = cursor + timedelta(days=period)

    # 3. Daily discretionary burn rate from recent profile
    profile = spending_profile(db_path, account_id, window_months=3)
    window_days = max((profile.window_end - profile.window_start).days, 1)
    daily_discretionary = profile.discretionary_total / window_days

    # Walk the window day by day applying events + discretionary burn.
    events_by_date: dict[date, list[CashflowEvent]] = {}
    for ev in events:
        events_by_date.setdefault(ev.on_date, []).append(ev)

    projection: list[tuple[date, float]] = []
    balance = start_balance
    cursor = start_date
    for _ in range(days + 1):
        for ev in events_by_date.get(cursor, []):
            balance += ev.amount
        balance -= daily_discretionary
        projection.append((cursor, round(balance, 2)))
        cursor = cursor + timedelta(days=1)

    # Record the implicit discretionary burn as a single summary event so
    # the caller can see what drove the decline.
    events.append(
        CashflowEvent(
            on_date=start_date,
            amount=-daily_discretionary * (days + 1),
            label="discretionary burn (total over window)",
            kind="discretionary",
        )
    )
    events.sort(key=lambda e: e.on_date)

    return CashflowForecast(
        start_date=start_date,
        start_balance=round(start_balance, 2),
        daily_projection=projection,
        events=events,
        assumptions={
            "daily_discretionary_burn": round(daily_discretionary, 2),
            "spending_profile_window_days": window_days,
            "wage_detected": wage is not None,
            "recurring_outflows_detected": len(outgoings),
        },
    )


# ---------------------------------------------------------------------------
# Frequency and classification helpers
# ---------------------------------------------------------------------------


_FREQUENCY_DAYS: dict[str, int] = {
    "WEEKLY": 7,
    "FORTNIGHTLY": 14,
    "MONTHLY": 30,
    "QUARTERLY": 91,
    "YEARLY": 365,
}


def _classify_frequency(gaps_days: list[float]) -> tuple[str | None, float]:
    """Given inter-arrival gaps (in days), return a frequency label and
    a confidence ``0.0 - 1.0`` based on how tight the gaps are.
    """
    if not gaps_days:
        return None, 0.0
    median = statistics.median(gaps_days)
    spread = statistics.pstdev(gaps_days) if len(gaps_days) > 1 else 0.0

    def _conf(tolerance: float) -> float:
        return max(0.0, 1.0 - (spread / tolerance))

    if 5 <= median <= 9:
        return "WEEKLY", _conf(3.0)
    if 12 <= median <= 17:
        return "FORTNIGHTLY", _conf(4.0)
    if 26 <= median <= 34:
        return "MONTHLY", _conf(5.0)
    if 85 <= median <= 95:
        return "QUARTERLY", _conf(10.0)
    if 355 <= median <= 375:
        return "YEARLY", _conf(20.0)
    return None, 0.0


def _frequency_from_truelayer_code(code: str | None) -> str | None:
    """Map TrueLayer / UK OBIE standing order frequency codes to our enum.

    Real values look like ``EvryDay``, ``Wkly``, ``Fortntly``, ``Mnthly``,
    ``Qtly``, ``Yrly``. We substring-match case-insensitively.
    """
    if not code:
        return None
    c = code.upper()
    # Fortnightly must be checked before weekly — some codes contain 'WK'
    # as a substring of the fortnightly variant.
    if "FORTN" in c or "FRTNT" in c or "FTNT" in c:
        return "FORTNIGHTLY"
    if "WK" in c or "WEEK" in c:
        return "WEEKLY"
    if "MNTH" in c or "MONTH" in c:
        return "MONTHLY"
    if "QT" in c or "QUART" in c:
        return "QUARTERLY"
    if "YR" in c or "YEAR" in c or "ANNU" in c:
        return "YEARLY"
    return None


def _monthly_equivalent(amount: float, frequency: str) -> float:
    period = _FREQUENCY_DAYS.get(frequency, 30)
    return amount * (30.0 / period)


# Hand-rolled keyword rules for classifying outgoings. Imperfect but cheap,
# and easy to extend as you learn which labels show up in real data.
_OUTGOING_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("rent", ("RENT", "LETTING", "LANDLORD", "PROPERTY", "LETTINGS")),
    (
        "utility",
        (
            "WATER",
            "GAS",
            "ELECTRIC",
            "ENERGY",
            "BRITISH GAS",
            "THAMES WATER",
            "EDF",
            "OCTOPUS",
            "EON",
            "BULB",
            "COUNCIL TAX",
            "BROADBAND",
            "SKY",
            "BT ",
            "VIRGIN MEDIA",
        ),
    ),
    ("insurance", ("INSURANCE", "INSURER", "AVIVA", "DIRECT LINE", "ADMIRAL")),
    (
        "subscription",
        (
            "NETFLIX",
            "SPOTIFY",
            "DISNEY",
            "PRIME",
            "APPLE",
            "GYM",
            "MEMBERSHIP",
            "SUBSCRIPTION",
            "YOUTUBE",
            "PATREON",
        ),
    ),
    ("savings", ("SAVINGS", "ISA", "VANGUARD", "TRADING", "INVESTMENT")),
]


def _classify_outgoing(label: str) -> str:
    if not label:
        return "other"
    upper = label.upper()
    for category, keywords in _OUTGOING_RULES:
        if any(kw in upper for kw in keywords):
            return category
    return "other"


_NON_DISCRETIONARY = {
    "Bills and Utilities",
    "Rent",
    "Mortgage",
    "Insurance",
    "Groceries",
    "Transportation",
    "Healthcare",
}


def _top_level_classification(raw: str | None) -> str:
    """Return the top level of a TrueLayer hierarchical classification.

    TrueLayer stores classification as a JSON array like
    ``["Bills and Utilities", "Utilities"]``. We keep only the first
    element; missing / empty values become ``"Uncategorised"``.
    """
    if not raw:
        return "Uncategorised"
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return "Uncategorised"
    if isinstance(parsed, list) and parsed:
        return str(parsed[0])
    return "Uncategorised"


# ---------------------------------------------------------------------------
# Small parsing / label utilities
# ---------------------------------------------------------------------------


_LABEL_NOISE_PATTERNS = (
    re.compile(r"\bREF[:\s]*\S*", re.IGNORECASE),
    re.compile(r"\b\d{6,}\b"),                          # long digit runs
    re.compile(r"\b\d{1,2}[A-Z]{3}\d{0,4}\b"),          # date-ish tokens
    re.compile(r"\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b"), # explicit dates
)


def _normalise_label(description: str) -> str:
    """Collapse a transaction description into a cluster key.

    Strips reference numbers, dates, and trailing noise so that, e.g.,
    ``"ACME CORP PAYROLL REF:83472 12APR"`` and
    ``"ACME CORP PAYROLL REF:92014 14MAY"`` both normalise to
    ``"ACME CORP PAYROLL"``.
    """
    if not description:
        return ""
    s = description.upper()
    for pat in _LABEL_NOISE_PATTERNS:
        s = pat.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _parse_iso(value: str) -> datetime:
    """Parse an ISO-8601 string, normalising ``Z`` to ``+00:00``."""
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _parse_iso_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return _parse_iso(value).date()
    except ValueError:
        return None


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _iso_days_ago(days: int) -> str:
    return (_utc_now() - timedelta(days=days)).isoformat()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
