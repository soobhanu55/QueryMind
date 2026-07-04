"""
Lightweight schema description store.

Rather than introspecting the live Postgres catalog (or dumping the full DDL) on
every request, we keep a small, hand-curated JSON description of the tables,
columns, synonyms and sample values (db/schema_metadata.json). This is:
  - cheap to load (parsed once, cached in memory as a module-level singleton)
  - cheap to search (keyword/phrase overlap, no embedding model required)
  - the single source of truth the guardrail layer uses to allow-list tables/columns

`retrieve_relevant_tables` implements simple schema linking: score each table by
how many of its name/synonym/column/sample-value phrases appear in the question,
then expand one hop along foreign-key relationships so join paths aren't missed
(e.g. a question about "revenue by region" scores high on order_items/customers
but still needs `orders` to bridge the join).
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Set

_METADATA_PATH = os.path.join(os.path.dirname(__file__), "..", "db", "schema_metadata.json")


@dataclass
class TableInfo:
    name: str
    description: str
    synonyms: List[str]
    columns: Dict[str, dict]
    trigger_phrases: List[str] = field(default_factory=list)


class SchemaStore:
    def __init__(self, metadata_path: str = _METADATA_PATH):
        with open(metadata_path, "r", encoding="utf-8") as f:
            self._raw = json.load(f)

        self.tables: Dict[str, TableInfo] = {}
        for table_name, tdef in self._raw["tables"].items():
            phrases = set()
            phrases.add(table_name.replace("_", " "))
            for syn in tdef.get("synonyms", []):
                phrases.add(syn.lower())
            for col_name, cdef in tdef["columns"].items():
                phrases.add(col_name.replace("_", " "))
                for syn in cdef.get("synonyms", []):
                    phrases.add(syn.lower())
                for val in cdef.get("sample_values", []) or []:
                    phrases.add(val.lower())
            self.tables[table_name] = TableInfo(
                name=table_name,
                description=tdef.get("description", ""),
                synonyms=tdef.get("synonyms", []),
                columns=tdef["columns"],
                trigger_phrases=sorted(phrases, key=len, reverse=True),
            )

        self.relationships = self._raw.get("relationships", [])
        self.metric_hints = self._raw.get("metric_hints", {})

    def table_names(self) -> List[str]:
        return list(self.tables.keys())

    def allowed_columns(self, table: str) -> Set[str]:
        return set(self.tables[table].columns.keys()) if table in self.tables else set()

    def sample_values_index(self) -> Dict[str, List[tuple]]:
        """value(lowercased) -> list of (table.column, original_cased_value) it can appear in.

        The original casing is preserved separately because Postgres string comparison
        is case-sensitive -- a filter built from the lowercased matching key would never
        match real rows (e.g. 'enterprise' != 'Enterprise').
        """
        idx: Dict[str, List[tuple]] = {}
        for tname, tinfo in self.tables.items():
            for cname, cdef in tinfo.columns.items():
                for val in cdef.get("sample_values", []) or []:
                    idx.setdefault(val.lower(), []).append((f"{tname}.{cname}", val))
        return idx

    def score_tables(self, question: str) -> Dict[str, int]:
        q = f" {question.lower()} "
        q = re.sub(r"[^a-z0-9 ]", " ", q)
        scores: Dict[str, int] = {}
        for tname, tinfo in self.tables.items():
            score = 0
            for phrase in tinfo.trigger_phrases:
                p = re.sub(r"[^a-z0-9 ]", " ", phrase.lower())
                if p and f" {p} " in q:
                    score += len(p.split())  # multi-word phrase matches count more
            scores[tname] = score
        return scores

    def _relationship_neighbors(self, table: str) -> Set[str]:
        neighbors = set()
        for rel in self.relationships:
            left_table = rel["from"].split(".")[0]
            right_table = rel["to"].split(".")[0]
            if left_table == table:
                neighbors.add(right_table)
            if right_table == table:
                neighbors.add(left_table)
        return neighbors

    def retrieve_relevant_tables(self, question: str, top_k: int = 4, max_expanded: int = 6) -> List[str]:
        scores = self.score_tables(question)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        selected = [t for t, s in ranked[:top_k] if s > 0]

        if not selected:
            # Fall back to the two most central tables so the LLM still has something
            # concrete to ground on rather than the entire schema.
            selected = ["orders", "order_items"]

        expanded = set(selected)
        for t in list(selected):
            for neighbor in self._relationship_neighbors(t):
                if len(expanded) >= max_expanded:
                    break
                expanded.add(neighbor)

        # keep deterministic order: original ranking first, then expansions
        ordered = [t for t in [x[0] for x in ranked] if t in expanded]
        return ordered

    def render_schema_text(self, table_names: List[str]) -> str:
        lines = []
        for tname in table_names:
            tinfo = self.tables[tname]
            lines.append(f"TABLE {tname} -- {tinfo.description}")
            for cname, cdef in tinfo.columns.items():
                bits = [cdef["type"]]
                if cdef.get("pk"):
                    bits.append("PRIMARY KEY")
                if cdef.get("fk"):
                    bits.append(f"REFERENCES {cdef['fk']}")
                extra = ", ".join(bits)
                desc = cdef.get("description", "")
                samples = cdef.get("sample_values")
                sample_txt = f" [values: {', '.join(samples)}]" if samples else ""
                lines.append(f"  - {cname} ({extra}): {desc}{sample_txt}")
            lines.append("")
        rels = [
            r for r in self.relationships
            if r["from"].split(".")[0] in table_names and r["to"].split(".")[0] in table_names
        ]
        if rels:
            lines.append("RELATIONSHIPS:")
            for r in rels:
                lines.append(f"  - {r['from']} = {r['to']}")
        hints = self.metric_hints
        if hints:
            lines.append("\nMETRIC HINTS:")
            for k, v in hints.items():
                lines.append(f"  - {k}: {v}")
        return "\n".join(lines)

    def full_schema_text(self) -> str:
        return self.render_schema_text(self.table_names())


@lru_cache(maxsize=1)
def get_schema_store() -> SchemaStore:
    return SchemaStore()
