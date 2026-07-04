// k6 load test for the /query endpoint (alternative to locustfile.py).
//
// Run:
//   k6 run --vus 50 --duration 60s tests/load/k6_script.js
//   BASE_URL=http://localhost:8000 k6 run --vus 50 --duration 60s tests/load/k6_script.js
//
// Reports RPS, error rate, and p50/p95/p99 latency automatically in the k6 summary.

import http from "k6/http";
import { check, sleep } from "k6";
import { Trend } from "k6/metrics";

const BASE_URL = __ENV.BASE_URL || "http://localhost:8000";

const QUESTIONS = [
  "List all customers in the APAC region.",
  "Show all products in the Electronics category.",
  "List all orders with status cancelled.",
  "What is the total revenue by region?",
  "How many orders were cancelled?",
  "What is the average order value for Enterprise customers?",
  "Show the top 5 customers by revenue.",
  "What is the total revenue in 2025?",
  "Which employee generated the most revenue?",
  "How many payments failed?",
];

const REPEATED_QUESTIONS = QUESTIONS.slice(0, 5);

const queryLatency = new Trend("query_latency_ms", true);

export const options = {
  thresholds: {
    http_req_failed: ["rate<0.01"],
    query_latency_ms: ["p(95)<2000"],
  },
};

export default function () {
  const useRepeated = Math.random() < 0.7;
  const pool = useRepeated ? REPEATED_QUESTIONS : QUESTIONS;
  const question = pool[Math.floor(Math.random() * pool.length)];

  const res = http.post(
    `${BASE_URL}/query`,
    JSON.stringify({ question }),
    { headers: { "Content-Type": "application/json" } }
  );

  queryLatency.add(res.timings.duration);

  check(res, {
    "status is 200": (r) => r.status === 200,
    "has sql field": (r) => {
      try {
        return JSON.parse(r.body).sql !== undefined;
      } catch (e) {
        return false;
      }
    },
  });

  sleep(Math.random() * 1.3 + 0.2);
}
