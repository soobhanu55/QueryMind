"""
Locust load test for the /query endpoint.

Run headless, e.g.:
    locust -f tests/load/locustfile.py --host http://localhost:8000 \
        --headless -u 50 -r 10 -t 60s --csv=reports/load_after --csv-full-history

See tests/load/run_load_comparison.py for a scripted before/after run that also
parses the resulting CSVs into reports/load_test_report.md.
"""
import json
import random

from locust import HttpUser, between, task

# A representative mix of simple/join/aggregation questions (drawn from the labeled
# accuracy test set) so the load test exercises the same code paths -- schema
# retrieval, generation, guardrail check, execution, summarization -- as real usage.
QUESTIONS = [
    "List all customers in the APAC region.",
    "Show all products in the Electronics category.",
    "List all orders with status cancelled.",
    "Show all employees who are Sales Managers.",
    "List all payments that failed.",
    "Show distinct product names purchased by customers in the LATAM region.",
    "List distinct customer names who placed orders that were delivered.",
    "What is the total revenue by region?",
    "How many orders were cancelled?",
    "What is the average order value for Enterprise customers?",
    "What is the total quantity sold per product category?",
    "How many customers signed up in each region?",
    "Show the top 5 customers by revenue.",
    "What is the total revenue in 2025?",
    "Which employee generated the most revenue?",
    "What is the total revenue by customer segment?",
    "How many payments failed?",
    "What is the average discount percentage applied across all order items?",
]

# A small fixed subset repeated often simulates the "same dashboard question asked
# over and over" pattern that the schema/question caches are meant to absorb.
REPEATED_QUESTIONS = QUESTIONS[:5]


class AnalyticsUser(HttpUser):
    # Short wait time simulates a bursty, high-throughput client (dashboard
    # auto-refresh / batch analytics) rather than a single human clicking around --
    # this is what's needed to actually exhaust a small DB connection pool and make
    # the pooling/caching optimizations show up in the numbers.
    wait_time = between(0.05, 0.4)

    @task(7)
    def ask_repeated_question(self):
        question = random.choice(REPEATED_QUESTIONS)
        self._post_query(question)

    @task(3)
    def ask_varied_question(self):
        question = random.choice(QUESTIONS)
        self._post_query(question)

    def _post_query(self, question: str):
        with self.client.post(
            "/query",
            json={"question": question},
            catch_response=True,
            name="/query",
        ) as response:
            if response.status_code != 200:
                response.failure(f"status={response.status_code} body={response.text[:200]}")
                return
            try:
                data = response.json()
                if "sql" not in data:
                    response.failure("missing 'sql' in response")
            except json.JSONDecodeError:
                response.failure("invalid JSON response")
