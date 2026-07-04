"""Builds the schema-constrained prompt sent to the SQL-generation LLM.

Only the tables retrieved as relevant by app.schema_store (not the full database
schema) are embedded, keeping the prompt small and keeping the model's attention on
real tables/columns rather than the entire catalog.
"""
from __future__ import annotations

SYSTEM_PROMPT = """You are a PostgreSQL expert that translates analytics questions into a single \
read-only SQL query.

Rules:
- Use ONLY the tables and columns listed under SCHEMA below. Never invent a table or column name.
- Generate exactly one SELECT statement (CTEs are fine). Never generate INSERT/UPDATE/DELETE/DDL \
or any statement that modifies data.
- Always qualify ambiguous columns with their table name/alias.
- Prefer explicit JOINs over implicit comma joins.
- If the question requires a metric covered by METRIC HINTS, follow that guidance.
- Respond by calling the `generate_sql` tool exactly once with the SQL, a confidence score between \
0 and 1 reflecting how certain you are the SQL answers the question correctly, and a one-sentence \
explanation of the query's logic.
"""

USER_PROMPT_TEMPLATE = """SCHEMA:
{schema_text}

QUESTION:
{question}
"""


def build_prompt(question: str, schema_text: str) -> tuple[str, str]:
    return SYSTEM_PROMPT, USER_PROMPT_TEMPLATE.format(schema_text=schema_text, question=question)


GENERATE_SQL_TOOL = {
    "name": "generate_sql",
    "description": "Return the generated read-only SQL query along with a confidence score and explanation.",
    "input_schema": {
        "type": "object",
        "properties": {
            "sql": {"type": "string", "description": "A single read-only PostgreSQL SELECT statement."},
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "Confidence that this SQL correctly answers the question.",
            },
            "explanation": {
                "type": "string",
                "description": "One-sentence, plain-language explanation of what the query does.",
            },
        },
        "required": ["sql", "confidence", "explanation"],
    },
}
