"""
Standalone Streamlit UI for the Enterprise Text-to-SQL Analytics Agent.

This is a restyled version of the original streamlit_app.py: same pipeline
(schema retrieval -> SQL generation -> guardrail -> execution -> summary) via
asyncio.run() per interaction, but with:
  - the Sage Ice / Electric Coral / Midnight Obsidian palette (.streamlit/config.toml
    + injected CSS below)
  - generated SQL hidden from the UI entirely (guardrail still runs on it server-side)
  - a "winner" highlight callout for the answer, with metrics shown above it
  - example questions collapsed into a click-to-open dock (st.expander)
  - a separate History page (session-state view toggle) instead of a sidebar list
  - an English / Deutsch label toggle for the UI chrome
"""
import os

import streamlit as st

st.set_page_config(page_title="Text-to-SQL Analytics Agent", page_icon="\U0001f4ca", layout="wide")

_secrets_error = None
try:
    _secrets = dict(st.secrets)
except FileNotFoundError:
    _secrets = {}
except Exception as _exc:  # noqa: BLE001
    _secrets = {}
    _secrets_error = repr(_exc)

for _key in (
    "DATABASE_URL", "REDIS_URL", "CACHE_ENABLED",
    "LLM_PROVIDER", "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL",
    "GEMINI_API_KEY", "GEMINI_MODEL",
    "MAX_RESULT_ROWS", "QUERY_TIMEOUT_SECONDS",
):
    if _key in _secrets:
        os.environ[_key] = str(_secrets[_key])

import asyncio
import time

import structlog
from dotenv import load_dotenv

load_dotenv()

logger = structlog.get_logger(__name__)

from app import cache as cache_module  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.db import dispose_engine  # noqa: E402
from app.cache import get_cache, question_cache_key, schema_cache_key  # noqa: E402
from app.executor import QueryExecutionError, execute_query  # noqa: E402
from app.guardrails.rules import check_sql  # noqa: E402
from app.llm import SQLGenerationResult, get_provider  # noqa: E402
from app.schema_store import get_schema_store  # noqa: E402
from app.summarizer import summarize  # noqa: E402

EXAMPLE_QUESTIONS = [
    "How many orders were cancelled?",
    "Show the top 5 customers by revenue.",
    "List all customers in the APAC region.",
    "Which employee generated the most revenue?",
    "What is the average order value for Enterprise customers?",
]

STRINGS = {
    "en": {
        "title": "\U0001f4ca Enterprise Text-to-SQL Analytics Agent",
        "try_question": "Try a question",
        "history": "History",
        "view_history": "View history \u2192",
        "back": "\u2190 Back",
        "no_history": "No queries run yet.",
        "question_label": "Ask an analytics question about the sample sales/orders database",
        "question_placeholder": "What is the total revenue by region?",
        "run": "Run query",
        "guardrail_caption": "rule-based, read-only SELECT only",
        "blocked_prefix": "\U0001f6ab Blocked by guardrail:",
        "blocked_note": "The generated SQL never reached the database \u2014 this is the guardrail layer working as intended.",
        "confidence": "Confidence", "rows": "Rows returned", "exec_time": "Execution time", "cached": "Cached generation",
        "show_details": "Show details", "explanation": "Explanation:", "timing": "Timing:",
        "truncated": "Results truncated at the row safety cap.",
        "spinner": "Generating SQL and querying the database...",
        "yes": "yes", "no": "no",
    },
    "de": {
        "title": "\U0001f4ca Enterprise Text-zu-SQL Analytics-Agent",
        "try_question": "Frage ausprobieren",
        "history": "Verlauf",
        "view_history": "Verlauf anzeigen \u2192",
        "back": "\u2190 Zur\u00fcck",
        "no_history": "Noch keine Abfragen ausgef\u00fchrt.",
        "question_label": "Stellen Sie eine Analysefrage zur Beispiel-Verkaufsdatenbank",
        "question_placeholder": "Wie hoch ist der Gesamtumsatz nach Region?",
        "run": "Abfrage ausf\u00fchren",
        "guardrail_caption": "regelbasiert, nur lesende SELECT-Abfragen",
        "blocked_prefix": "\U0001f6ab Von der Guardrail blockiert:",
        "blocked_note": "Das generierte SQL hat die Datenbank nie erreicht \u2014 die Guardrail-Schicht funktioniert wie vorgesehen.",
        "confidence": "Konfidenz", "rows": "Zeilen zur\u00fcckgegeben", "exec_time": "Ausf\u00fchrungszeit", "cached": "Zwischengespeichert",
        "show_details": "Details anzeigen", "explanation": "Erkl\u00e4rung:", "timing": "Zeiten:",
        "truncated": "Ergebnisse bei der Sicherheitsgrenze abgeschnitten.",
        "spinner": "SQL wird generiert und die Datenbank abgefragt...",
        "yes": "ja", "no": "nein",
    },
}

CSS = """
<style>
  :root {
    --bg: #D2E3D8; --surface: #FFFFFF; --text: #0D1B1E; --accent: #FF5A36;
  }
  .stApp { background: var(--bg); }
  h1, h2, h3 { color: var(--text) !important; font-weight: 600; }
  .stButton > button {
    border-radius: 8px; border: 1px solid var(--text); color: var(--text);
    background: var(--surface); font-weight: 500;
  }
  .stButton > button:hover { border-color: var(--accent); color: var(--accent); }
  .stButton > button[kind="primary"] {
    background: var(--accent); border-color: var(--accent); color: #fff;
  }
  div[data-testid="stMetric"] {
    background: var(--surface); border-radius: 10px; padding: 12px 14px;
    border: 1px solid rgba(13,27,30,0.08);
  }
  .sqla-winner {
    background: var(--surface); border: 1px solid var(--accent); border-radius: 12px;
    padding: 28px; text-align: center; margin: 18px 0;
  }
  .sqla-winner .kicker {
    display: inline-block; font-size: 11px; letter-spacing: 0.1em; text-transform: uppercase;
    color: var(--accent); border: 1px solid var(--accent); border-radius: 999px; padding: 3px 10px; margin-bottom: 10px;
  }
  .sqla-winner .value { font-size: 20px; font-weight: 600; color: var(--text); }
  .sqla-blocked {
    background: #fff; border: 1px solid var(--accent); border-radius: 10px; padding: 16px 18px; margin: 10px 0;
  }
</style>
"""


async def run_pipeline(question: str) -> dict:
    settings = get_settings()
    store = get_schema_store()
    cache = await get_cache()

    relevant_tables = store.retrieve_relevant_tables(question)
    schema_key = schema_cache_key(relevant_tables)
    schema_text = await cache.get_json(schema_key)
    if schema_text is None:
        schema_text = store.render_schema_text(relevant_tables)
        await cache.set_json(schema_key, schema_text, settings.schema_cache_ttl_seconds)

    q_key = question_cache_key(question)
    cached_generation = await cache.get_json(q_key)
    gen_start = time.perf_counter()
    cached = False
    if cached_generation is not None:
        generation = SQLGenerationResult(**cached_generation)
        cached = True
    else:
        provider = get_provider()
        generation = await provider.generate(question, schema_text)
        await cache.set_json(
            q_key,
            {"sql": generation.sql, "confidence": generation.confidence, "explanation": generation.explanation},
            settings.question_cache_ttl_seconds,
        )
    generation_ms = (time.perf_counter() - gen_start) * 1000

    guardrail_result = check_sql(
        generation.sql, allowed_tables=store.table_names(), row_limit_cap=settings.max_result_rows, question=question
    )
    if not guardrail_result.allowed:
        return {
            "blocked": True,
            "reason": guardrail_result.reason,
            "category": guardrail_result.blocked_category,
            "attempted_sql": generation.sql,
        }

    result = await execute_query(guardrail_result.sanitized_sql, row_cap=settings.max_result_rows)
    return {
        "blocked": False,
        "sql": guardrail_result.sanitized_sql,
        "confidence": generation.confidence,
        "explanation": generation.explanation,
        "columns": result.columns,
        "rows": result.rows,
        "row_count": result.row_count,
        "truncated": result.truncated,
        "summary": summarize(result),
        "cached": cached,
        "generation_ms": generation_ms,
        "guardrail_ms": guardrail_result.check_duration_ms,
        "execution_ms": result.execution_ms,
    }


async def run_and_cleanup(question: str) -> dict:
    try:
        return await run_pipeline(question)
    finally:
        try:
            await dispose_engine()
        except Exception as exc:  # noqa: BLE001
            logger.warning("db_engine_dispose_failed", error=str(exc))
        try:
            if cache_module._cache_client is not None:
                await cache_module._cache_client.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("cache_client_close_failed", error=str(exc))
        finally:
            cache_module._cache_client = None


def render_main(t, settings):
    st.title(t["title"])
    st.caption(f"LLM provider: `{settings.llm_provider}`  |  Guardrail: {t['guardrail_caption']}  |  Row cap: {settings.max_result_rows}")

    with st.expander(f"\U0001f527 {t['try_question']}"):
        for q in EXAMPLE_QUESTIONS:
            if st.button(q, key=f"ex_{q}", use_container_width=True):
                st.session_state["question"] = q

    col_q, col_hist = st.columns([5, 1])
    with col_hist:
        st.write("")
        if st.button(t["view_history"], use_container_width=True):
            st.session_state["view"] = "history"
            st.rerun()

    question = st.text_area(
        t["question_label"], key="question", placeholder=t["question_placeholder"], height=80,
    )
    run_clicked = st.button(t["run"], type="primary")

    if run_clicked and question.strip():
        with st.spinner(t["spinner"]):
            try:
                output = asyncio.run(run_and_cleanup(question))
            except QueryExecutionError as exc:
                st.error(f"Query execution failed: {exc}")
                return
            except Exception as exc:  # noqa: BLE001
                st.error(f"Unexpected error: {exc}")
                return

        st.session_state.setdefault("history", [])
        st.session_state["history"].insert(0, {"question": question, "blocked": output["blocked"]})
        st.session_state["history"] = st.session_state["history"][:20]

        if output["blocked"]:
            st.markdown(
                f'<div class="sqla-blocked"><strong>{t["blocked_prefix"]}</strong> {output["reason"]}'
                f' <code>({output["category"]})</code><br><span style="opacity:0.7;font-size:13px">{t["blocked_note"]}</span></div>',
                unsafe_allow_html=True,
            )
            return

        st.markdown(
            f'<div class="sqla-winner"><span class="kicker">Result</span><div class="value">{output["summary"]}</div></div>',
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(t["confidence"], f"{output['confidence']:.0%}")
        c2.metric(t["rows"], output["row_count"])
        c3.metric(t["exec_time"], f"{output['execution_ms']:.1f} ms")
        c4.metric(t["cached"], t["yes"] if output["cached"] else t["no"])

        st.dataframe([dict(zip(output["columns"], row)) for row in output["rows"]], use_container_width=True)

        if output["truncated"]:
            st.warning(t["truncated"])

        with st.expander(t["show_details"]):
            st.write(f"**{t['explanation']}** {output['explanation']}")
            st.write(
                f"**{t['timing']}** generation={output['generation_ms']:.3f}ms, "
                f"guardrail={output['guardrail_ms']:.4f}ms, execution={output['execution_ms']:.1f}ms"
            )


def render_history(t):
    if st.button(t["back"]):
        st.session_state["view"] = "main"
        st.rerun()
    st.header(t["history"])
    history = st.session_state.get("history", [])
    if not history:
        st.write(t["no_history"])
        return
    for i, h in enumerate(history):
        tag = "\U0001f6ab" if h["blocked"] else "\u2705"
        if st.button(f"{tag}  {h['question']}", key=f"hist_{i}", use_container_width=True):
            st.session_state["question"] = h["question"]
            st.session_state["view"] = "main"
            st.rerun()


def main():
    settings = get_settings()
    st.markdown(CSS, unsafe_allow_html=True)

    lang = st.sidebar.radio("Language / Sprache", ["English", "Deutsch"], horizontal=True)
    t = STRINGS["de"] if lang == "Deutsch" else STRINGS["en"]

    st.session_state.setdefault("view", "main")

    if _secrets_error:
        st.sidebar.error(f"st.secrets error: {_secrets_error}")

    if st.session_state["view"] == "history":
        render_history(t)
    else:
        render_main(t, settings)


if __name__ == "__main__":
    main()
