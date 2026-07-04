"""
Deterministic, offline rule-based NL2SQL engine used when llm_provider=mock.

This exists so the full pipeline -- prompt construction, guardrails, execution,
accuracy benchmarking, safety benchmarking, load testing -- is reproducible by
anyone cloning this repo with zero API cost and zero external dependency on an LLM
being available. It is a genuine (if small) semantic parser: it extracts an
aggregation function, a target metric, a group-by dimension, filters, and
top-N/ordering from the question text using the schema metadata's synonyms/sample
values for grounding, then assembles SQL from a small join-graph builder. It is not
a lookup table of the specific benchmark questions -- swapping in AnthropicProvider
(app/llm/anthropic_provider.py) via ANTHROPIC_API_KEY is what a production
deployment should do; this provider's accuracy score is a floor, not the ceiling
the 98%-aggregation-accuracy target should be read against (see README).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from app.llm.base import NL2SQLProvider, SQLGenerationResult
from app.schema_store import get_schema_store

ALIASES = {
    "orders": "o",
    "customers": "c",
    "employees": "e",
    "order_items": "oi",
    "products": "p",
    "payments": "pay",
}

# order matters: order_items must be joined before products; orders is the hub.
JOIN_ORDER = ["orders", "order_items", "products", "customers", "employees", "payments"]

JOIN_SQL = {
    "order_items": "JOIN order_items oi ON oi.order_id = o.order_id",
    "products": "JOIN products p ON p.product_id = oi.product_id",
    "customers": "JOIN customers c ON c.customer_id = o.customer_id",
    # employee_id/payments are nullable FKs on orders, but an INNER join (matching the
    # convention used throughout this project's reference SQL) is what "revenue by
    # employee" / "orders by payment method" style questions actually want: attribute
    # metrics only to known reps/payments rather than adding a spurious NULL bucket
    # that silently absorbs every unassigned order.
    "employees": "JOIN employees e ON e.employee_id = o.employee_id",
    "payments": "JOIN payments pay ON pay.order_id = o.order_id",
}

REVENUE_EXPR = "oi.quantity * oi.unit_price * (1 - oi.discount_pct / 100.0)"

ORDER_CONTEXT_RE = re.compile(
    r"\border(s|ed)?\b|\bpurchase(d|s)?\b|\bsold\b|\bplaced\b|\bbought\b|\btransaction"
)

# (regex, dimension sql expr, alias, tables required)
GROUP_BY_PATTERNS = [
    (re.compile(r"\bby (customer )?regions?\b|\bper region\b|\beach region\b|\bacross regions\b"),
     "c.region", "region", {"customers"}),
    (re.compile(r"\bby (?:\w+\s+)?categor(y|ies)\b|\bper (?:\w+\s+)?categor(y|ies)\b|"
                r"\beach (?:\w+\s+)?categor(y|ies)\b|\bacross categories\b"),
     "p.category", "category", {"order_items", "products"}),
    (re.compile(r"\bby segments?\b|\bper segment\b|\beach segment\b|\bcustomer segment\b"),
     "c.segment", "segment", {"customers"}),
    (re.compile(r"\bby status\b|\bper status\b|\beach status\b|\border status\b"),
     "o.status", "status", set()),
    (re.compile(r"\bby channel\b|\bper channel\b|\beach channel\b|\bsales channel\b"),
     "o.channel", "channel", set()),
    (re.compile(r"\bby payment method\b|\bper payment method\b|\beach payment method\b"),
     "pay.method", "method", {"payments"}),
    (re.compile(r"\bby (sales ?rep|employee|rep)s?\b|\bper (sales ?rep|employee|rep)\b|\beach (sales ?rep|employee|rep)\b"),
     "e.name", "employee_name", {"employees"}),
    (re.compile(r"\bby customer\b|\bper customer\b|\beach customer\b"),
     "c.name", "customer_name", {"customers"}),
    (re.compile(r"\bby product\b|\bper product\b|\beach product\b"),
     "p.name", "product_name", {"order_items", "products"}),
    (re.compile(r"\bby month\b|\bmonthly\b|\bper month\b|\beach month\b"),
     "date_trunc('month', o.order_date)", "month", set()),
    (re.compile(r"\bby year\b|\bper year\b|\beach year\b|\byearly\b"),
     "date_trunc('year', o.order_date)", "year", set()),
]

ID_COLUMN = {"customers": "c.customer_id", "employees": "e.employee_id", "products": "p.product_id"}

# Fallback group-by inference for "top N <entity>" / "which <entity> ..." phrasing that
# doesn't use an explicit "by X" clause, e.g. "top 5 customers by revenue" or
# "which employee generated the most revenue".
ENTITY_NOUN_GROUPBY = [
    (re.compile(r"\bproduct categor(y|ies)\b|\bcategor(y|ies)\b"), "p.category", "category", {"order_items", "products"}),
    (re.compile(r"\bemployees?\b|\bsales ?reps?\b"), "e.name", "employee_name", {"employees"}),
    (re.compile(r"\bcustomers?\b"), "c.name", "customer_name", {"customers"}),
    (re.compile(r"\bproducts?\b"), "p.name", "product_name", {"order_items", "products"}),
    (re.compile(r"\bregions?\b"), "c.region", "region", {"customers"}),
]

TOPN_RE = re.compile(r"\btop\s+(\d+)\b")
WORD_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
                "eight": 8, "nine": 9, "ten": 10}
TOPN_WORD_RE = re.compile(r"\btop\s+(" + "|".join(WORD_NUMBERS) + r")\b")
YEAR_RE = re.compile(r"\b(20\d{2})\b")
NUMERIC_CMP_RE = re.compile(r"\b(greater than|more than|over|above|at least|less than|under|below|at most)\s+\$?(\d+(?:\.\d+)?)")
CMP_MAP = {
    "greater than": ">", "more than": ">", "over": ">", "above": ">",
    "at least": ">=", "less than": "<", "under": "<", "below": "<", "at most": "<=",
}


@dataclass
class _Plan:
    tables: set = field(default_factory=set)
    where: list = field(default_factory=list)
    having: list = field(default_factory=list)
    confidence: float = 0.5


def _build_from_clause(tables: set) -> str:
    ordered = [t for t in JOIN_ORDER if t in tables]
    if not ordered:
        ordered = ["orders"]
    if "orders" not in tables:
        # standalone single/bridged-table query with no need for the orders hub
        # (e.g. "products" alone, or "order_items"+"products" without orders/dates)
        t0 = ordered[0]
        parts = [f"FROM {t0} {ALIASES[t0]}"]
        for t in ordered[1:]:
            parts.append(JOIN_SQL[t])
        return " ".join(parts)
    parts = ["FROM orders o"]
    for t in ordered:
        if t == "orders":
            continue
        parts.append(JOIN_SQL[t])
    return " ".join(parts)


def _needs_orders_hub(q: str, metric_tables: set, gb_tables: set, plan: _Plan) -> bool:
    if metric_tables & {"order_items", "payments"}:
        return True
    if gb_tables is not None and not gb_tables and gb_tables != set():
        pass
    if "orders" in plan.tables:
        return True
    if ORDER_CONTEXT_RE.search(q):
        return True
    return False


def _detect_agg(q: str) -> Optional[str]:
    if re.search(r"\bhow many\b|\bnumber of\b|\bcount of\b", q):
        return "count"
    if re.search(r"\btotal\b|\bsum of\b", q):
        return "sum"
    if re.search(r"\baverage\b|\bavg\b|\bmean\b", q):
        return "avg"
    return None


def _detect_topn(q: str) -> Optional[int]:
    m = TOPN_RE.search(q)
    if m:
        return int(m.group(1))
    m = TOPN_WORD_RE.search(q)
    if m:
        return WORD_NUMBERS[m.group(1)]
    return None


def _detect_superlative(q: str) -> Optional[str]:
    if re.search(r"\bhighest\b|\bmost\b|\bmaximum\b|\bbiggest\b|\blargest\b", q):
        return "desc"
    if re.search(r"\blowest\b|\bleast\b|\bminimum\b|\bsmallest\b|\bcheapest\b", q):
        return "asc"
    return None


_COUNT_ENTITY_TABLE = [
    (re.compile(r"\bcustomers?\b"), ("COUNT(*)", "customer_count", {"customers"}, "count", "customers")),
    (re.compile(r"\bemployees?\b|\bsales ?reps?\b"), ("COUNT(*)", "employee_count", {"employees"}, "count", "employees")),
    (re.compile(r"\bproducts?\b"), ("COUNT(*)", "product_count", {"products"}, "count", "products")),
    (re.compile(r"\bpayments?\b"), ("COUNT(pay.payment_id)", "payment_count", {"payments"}, "count", "payments")),
]

# The noun immediately after "how many"/"number of"/"count of" is the thing actually
# being counted -- e.g. in "how many orders does each sales rep have", "orders" is the
# head noun even though "sales rep" (the group-by dimension) also matches an entity
# pattern. Scanning the whole sentence for any entity word (rather than just the head
# noun) would wrongly count reps instead of orders in exactly that phrasing.
_HEAD_NOUN_RE = re.compile(r"\b(?:how many|number of|count of)\s+([a-z]+)")


def _agg_expr(agg: Optional[str], default: str, inner: str) -> str:
    """Builds e.g. AVG(x) vs SUM(x) from the actually-detected aggregation word,
    instead of hard-coding one function per metric regardless of what was asked."""
    return f"{(agg or default).upper()}({inner})"


# regex, alias/std-table, standalone table name (None if metric inherently needs the hub)
def _detect_metric(
    q: str, agg: Optional[str], gb_tables: frozenset = frozenset()
) -> tuple[str, str, set, str, Optional[str]]:
    """Returns (select_sql, alias, needed_tables, resolved_agg, standalone_table)."""
    if re.search(r"\baverage order value\b|\bavg order value\b|\baov\b", q):
        return (f"SUM({REVENUE_EXPR}) / NULLIF(COUNT(DISTINCT o.order_id), 0)",
                "avg_order_value", {"order_items"}, "computed", None)

    # An explicit "how many <entity>" should count that specific head-noun entity even
    # if some other metric-ish word (e.g. "paid") also appears in the sentence as a
    # status adjective rather than a metric noun ("how many payments were paid
    # successfully"). Only the head noun counts -- not just any entity word anywhere in
    # the sentence -- so "how many orders does each sales rep have" counts orders, not
    # reps, even though "sales rep" also matches an entity pattern (as the GROUP BY
    # dimension, not the thing being counted).
    if agg == "count":
        head_match = _HEAD_NOUN_RE.search(q)
        head = f" {head_match.group(1)} " if head_match else ""
        for pattern, result in _COUNT_ENTITY_TABLE:
            if pattern.search(head):
                return result

    if re.search(r"\brevenue\b|\bmoney (made|earned)\b|\bgross\b", q) or re.search(
        r"\bsales\b(?!\s*(rep|reps|manager|managers|team|person|people|executive|executives))", q
    ):
        return (_agg_expr(agg, "sum", REVENUE_EXPR), "total_revenue", {"order_items"}, agg or "sum", None)
    if re.search(r"\bpaid\b|\bcollected\b|\bpayments? (amount|total|received)\b", q):
        return (_agg_expr(agg, "sum", "pay.amount"), "total_paid", {"payments"}, agg or "sum", "payments")
    if re.search(r"\bdiscount\b", q):
        return (_agg_expr(agg, "avg", "oi.discount_pct"), "avg_discount_pct", {"order_items"}, agg or "avg", None)
    if re.search(r"\bquantity\b|\bunits sold\b|\bitems sold\b|\bnumber of (items|units)\b", q):
        return (_agg_expr(agg, "sum", "oi.quantity"), "total_quantity", {"order_items"}, agg or "sum", None)
    if re.search(r"\bprice\b|\bunit price\b|\bcost\b", q) and "product" in q:
        return (_agg_expr(agg, "avg", "p.unit_price"), "avg_price", {"products"}, agg or "avg", "products")

    for pattern, result in _COUNT_ENTITY_TABLE:
        if pattern.search(q) and not (result[2] & gb_tables):
            return result
    # default: orders/sales/purchases/transactions as the counted entity
    return ("COUNT(DISTINCT o.order_id)", "order_count", set(), "count", "orders")


def _apply_categorical_filters(q: str, plan: _Plan) -> set:
    """Returns the set of (table, column) pairs that received an equality filter."""
    store = get_schema_store()
    value_index = store.sample_values_index()
    table_scores = store.score_tables(q)
    filtered_columns = set()
    for value_lower, locations in value_index.items():
        # Schema sample values are stored in whatever form the DB uses (e.g. the
        # snake_case 'bank_transfer'), but a natural-language question says "bank
        # transfer" -- match either spelling while still emitting the DB's exact form
        # as the SQL literal.
        spaced = value_lower.replace("_", " ")
        pattern = re.compile(r"\b(?:" + re.escape(value_lower) + "|" + re.escape(spaced) + r")s?\b")
        if not pattern.search(q):
            continue
        if len(locations) > 1:
            locations = sorted(locations, key=lambda loc: table_scores.get(loc[0].split(".")[0], 0), reverse=True)
        table_col, original_value = locations[0]
        table, col = table_col.split(".")
        alias = ALIASES[table]
        plan.tables.add(table)
        # Postgres string comparison is case-sensitive -- use the schema's original
        # casing for the literal, not the lowercased text we matched against.
        plan.where.append(f"{alias}.{col} = '{original_value}'")
        filtered_columns.add((table, col))
    return filtered_columns


def _apply_date_filters(q: str, plan: _Plan) -> bool:
    added = False
    m = re.search(r"\bin (20\d{2})\b|\bduring (20\d{2})\b|\bfor (20\d{2})\b", q)
    if m:
        year = next(g for g in m.groups() if g)
        plan.where.append(f"EXTRACT(YEAR FROM o.order_date) = {year}")
        return True
    m = YEAR_RE.search(q)
    if m:
        plan.where.append(f"EXTRACT(YEAR FROM o.order_date) = {m.group(1)}")
        return True
    m = re.search(r"\blast (\d+) days\b", q)
    if m:
        plan.where.append(f"o.order_date >= CURRENT_DATE - INTERVAL '{int(m.group(1))} days'")
        return True
    if re.search(r"\blast 30 days\b|\bpast month\b", q):
        plan.where.append("o.order_date >= CURRENT_DATE - INTERVAL '30 days'")
        added = True
    elif re.search(r"\blast quarter\b|\bpast 90 days\b", q):
        plan.where.append("o.order_date >= CURRENT_DATE - INTERVAL '90 days'")
        added = True
    elif re.search(r"\blast year\b|\bpast 12 months\b", q):
        plan.where.append("o.order_date >= CURRENT_DATE - INTERVAL '1 year'")
        added = True
    return added


def _apply_numeric_filters(q: str, plan: _Plan, metric_expr: str) -> None:
    m = NUMERIC_CMP_RE.search(q)
    if not m:
        return
    cmp_op = CMP_MAP[m.group(1)]
    value = m.group(2)
    if re.search(r"\bitems?\b|\bunits?\b|\bquantity\b", q):
        plan.where.append(f"oi.quantity {cmp_op} {value}")
        plan.tables.add("order_items")
    elif re.search(r"\bprice\b", q):
        plan.tables.add("products")
        plan.where.append(f"p.unit_price {cmp_op} {value}")
    else:
        # assume it constrains the aggregated metric itself -> HAVING clause
        plan.having.append(f"{metric_expr} {cmp_op} {value}")


SIMPLE_LOOKUP_ROW_CAP = 500  # matches settings.max_result_rows / guardrail default cap

# Reuses the same column-vocabulary approach as GROUP_BY_PATTERNS/ENTITY_NOUN_GROUPBY,
# but for detecting which specific column(s) a non-aggregation "list/show" question
# wants projected (e.g. "distinct payment methods", "customer names and regions"),
# instead of always falling back to a blind SELECT *.
PROJECTION_PATTERNS = [
    (re.compile(r"\bcustomer names?\b"), "c.name", "customer_name", {"customers"}),
    (re.compile(r"\bemployee names?\b|\bsales ?rep names?\b"), "e.name", "employee_name", {"employees"}),
    (re.compile(r"\bproduct names?\b"), "p.name", "product_name", {"order_items", "products"}),
    (re.compile(r"\b(product )?categor(y|ies)\b"), "p.category", "category", {"order_items", "products"}),
    (re.compile(r"\bregions?\b"), "c.region", "region", {"customers"}),
    (re.compile(r"\bsegments?\b"), "c.segment", "segment", {"customers"}),
    (re.compile(r"\border dates?\b"), "o.order_date", "order_date", set()),
    (re.compile(r"\border status(es)?\b"), "o.status", "status", set()),
    (re.compile(r"\border channels?\b|\bsales channels?\b"), "o.channel", "channel", set()),
    (re.compile(r"\bpayment methods?\b"), "pay.method", "method", {"payments"}),
    (re.compile(r"\bpayment status(es)?\b"), "pay.status", "payment_status", {"payments"}),
]


_ALIAS_TO_TABLE = {v: k for k, v in ALIASES.items()}


def _detect_projection(q: str, filtered_columns: set) -> tuple[Optional[list[str]], set]:
    """Finds specific columns named in the question, in the order they appear.

    A mention is skipped if that exact (table, column) already received an equality
    filter elsewhere in the question -- "customers in the APAC region" is a filter
    on region, not a request to list the region column, whereas "list customer
    regions" (no specific region named) really is asking to project it.
    """
    matches = []
    seen_alias = set()
    for pattern, col_sql, alias, tables in PROJECTION_PATTERNS:
        if "." not in col_sql:
            continue
        col_alias, col_name = col_sql.split(".", 1)
        table = _ALIAS_TO_TABLE.get(col_alias)
        if table and (table, col_name) in filtered_columns:
            continue
        m = pattern.search(q)
        if m and alias not in seen_alias:
            seen_alias.add(alias)
            matches.append((m.start(), f"{col_sql} AS {alias}", tables))
    if not matches:
        return None, set()
    matches.sort(key=lambda x: x[0])
    tables = set()
    for _, _, t in matches:
        tables |= t
    return [proj for _, proj, _ in matches], tables


def _handle_simple_lookup(q: str, store) -> Optional[SQLGenerationResult]:
    """Non-aggregation 'list/show/what is' queries, possibly spanning multiple joined tables."""
    # Filters first, on a scratch plan, so we know which columns are "filter mentions"
    # (e.g. "APAC region") before deciding what to project.
    scratch = _Plan()
    date_filter_added = _apply_date_filters(q, scratch)
    filtered_columns = _apply_categorical_filters(q, scratch)

    projection_cols, projection_tables = _detect_projection(q, filtered_columns)

    if projection_cols:
        plan = _Plan(tables=set(projection_tables) | scratch.tables, where=list(scratch.where))
    elif scratch.tables:
        # A filter grounded us in a specific table (e.g. "customers" via region='APAC') --
        # trust that over a generic relevance guess.
        plan = _Plan(tables=set(scratch.tables), where=list(scratch.where))
    else:
        primary = store.retrieve_relevant_tables(q, top_k=1)
        plan = _Plan(tables=set(primary) if primary else {"orders"})

    if date_filter_added:
        plan.tables.add("orders")

    # Bridging: products only connects to the rest of the graph through order_items;
    # everything else (customers/employees/payments/order_items) only connects to each
    # other through the orders hub. Add whichever bridge tables are missing so the
    # eventual FROM clause never references an alias that isn't actually joined in.
    if "products" in plan.tables and "order_items" not in plan.tables and len(plan.tables) > 1:
        plan.tables.add("order_items")
    if len(plan.tables) > 1 and not plan.tables <= {"order_items", "products"}:
        plan.tables.add("orders")

    limit = _detect_topn(q) or SIMPLE_LOOKUP_ROW_CAP
    superlative = _detect_superlative(q)

    order_clause = ""
    primary_table = next(iter(projection_tables), None) or (sorted(plan.tables)[0] if plan.tables else None)
    if superlative and primary_table == "products":
        order_clause = f" ORDER BY p.unit_price {'DESC' if superlative == 'desc' else 'ASC'}"

    from_clause = _build_from_clause(plan.tables)
    where_clause = f" WHERE {' AND '.join(plan.where)}" if plan.where else ""
    # DISTINCT avoids duplicate rows when a join fans out (e.g. one customer with many
    # orders) for what is conceptually a listing of distinct entities/combinations.
    select_list = ", ".join(projection_cols) if projection_cols else "*"
    sql = f"SELECT DISTINCT {select_list} {from_clause}{where_clause}{order_clause} LIMIT {limit}"
    confidence = 0.6 if projection_cols else (0.55 if plan.where else 0.4)
    explanation = f"Lists {'columns ' + select_list if projection_cols else 'rows'} from {', '.join(sorted(plan.tables))} matching the question's filters."
    return SQLGenerationResult(sql=sql, confidence=confidence, explanation=explanation)


class MockNL2SQLProvider(NL2SQLProvider):
    async def generate(self, question: str, schema_text: str) -> SQLGenerationResult:
        store = get_schema_store()
        q = " " + re.sub(r"[^a-z0-9$.\s]", " ", question.lower()) + " "
        q = re.sub(r"\s+", " ", q)

        agg = _detect_agg(q)
        topn = _detect_topn(q)
        superlative = _detect_superlative(q)
        is_which_what = bool(re.match(r"^\s*(which|what)\b", q.strip()))

        gb_sql, gb_alias, gb_tables = None, None, set()
        for pattern, sql_expr, alias, tables in GROUP_BY_PATTERNS:
            if pattern.search(q):
                gb_sql, gb_alias, gb_tables = sql_expr, alias, tables
                break

        aggregate_signal = agg is not None or topn is not None or superlative is not None or gb_sql is not None
        if not aggregate_signal:
            result = _handle_simple_lookup(q, store)
            if result is not None:
                return result

        metric_sql, metric_alias, metric_tables, resolved_agg, standalone_table = _detect_metric(
            q, agg, gb_tables=frozenset(gb_tables)
        )
        plan = _Plan(confidence=0.6)

        if gb_sql:
            plan.confidence += 0.15
        elif topn is not None or superlative is not None:
            # "top 5 customers by revenue" / "which employee generated the most revenue"
            for pattern, sql_expr, alias, tables in ENTITY_NOUN_GROUPBY:
                if pattern.search(q):
                    gb_sql, gb_alias, gb_tables = sql_expr, alias, tables
                    plan.confidence += 0.1
                    break

        date_filter_added = _apply_date_filters(q, plan)
        _apply_categorical_filters(q, plan)
        _apply_numeric_filters(q, plan, metric_sql)
        if plan.where or plan.having:
            plan.confidence += 0.1

        # "order_items" always tolerates the hub (INNER JOIN, 1+ item per order in this
        # schema, no fan-out risk); "payments" does NOT (nullable 0-or-1 relation to
        # orders) -- forcing it through a LEFT JOIN orders hub introduces a spurious
        # NULL group whenever grouping by a payments column with no order-level filter.
        # So "payments" alone should stay a standalone table unless something else
        # (a customer/employee attribute, a date filter, explicit order-context wording)
        # requires bridging through orders.
        needs_hub = (
            "order_items" in metric_tables
            or "order_items" in gb_tables
            or (gb_alias in {"status", "channel", "month", "year"})
            or date_filter_added
            or bool(ORDER_CONTEXT_RE.search(q))
            or "order_items" in plan.tables
            or bool(plan.tables & {"customers", "employees"})
        )

        if needs_hub:
            plan.tables.add("orders")
            plan.tables |= metric_tables
            plan.tables |= gb_tables
            if standalone_table in ID_COLUMN and metric_sql == "COUNT(*)":
                # joined across the hub, so COUNT(*) would double-count per matching
                # row -- switch to COUNT(DISTINCT id) of the entity actually being counted
                metric_sql = f"COUNT(DISTINCT {ID_COLUMN[standalone_table]})"
                plan.tables.add(standalone_table)
        else:
            plan.tables |= metric_tables
            plan.tables |= gb_tables
            if standalone_table:
                plan.tables.add(standalone_table)

        order_dir = superlative or ("desc" if topn else None)
        limit = topn or (1 if (superlative and (is_which_what or gb_sql)) else None)

        from_clause = _build_from_clause(plan.tables)
        where_clause = f" WHERE {' AND '.join(plan.where)}" if plan.where else ""
        having_clause = f" HAVING {' AND '.join(plan.having)}" if plan.having else ""

        if gb_sql:
            if order_dir or limit:
                order_clause = f" ORDER BY {metric_alias} {'DESC' if order_dir != 'asc' else 'ASC'}"
            else:
                order_clause = f" ORDER BY {gb_alias}"
            limit_clause = f" LIMIT {limit}" if limit else ""
            sql = (
                f"SELECT {gb_sql} AS {gb_alias}, {metric_sql} AS {metric_alias} "
                f"{from_clause}{where_clause} GROUP BY {gb_sql}{having_clause}{order_clause}{limit_clause}"
            )
        else:
            sql = f"SELECT {metric_sql} AS {metric_alias} {from_clause}{where_clause}{having_clause}"

        confidence = min(0.97, plan.confidence)
        explanation = (
            f"Computes {resolved_agg.upper()} over {metric_alias.replace('_', ' ')}"
            + (f", grouped by {gb_alias}" if gb_alias else "")
            + (f", filtered by {' and '.join(plan.where)}" if plan.where else "")
            + "."
        )
        return SQLGenerationResult(sql=sql, confidence=round(confidence, 2), explanation=explanation)
