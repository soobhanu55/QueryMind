"""
Generates a single .sql file (schema + synthetic seed data) that can be pasted
directly into a hosted Postgres provider's web SQL editor (e.g. Supabase's SQL
Editor), for environments where a direct asyncpg/psycopg connection from the local
machine isn't possible (e.g. no IPv6 route to a provider's IPv6-only direct-connect
endpoint).

Mirrors db/seed.py's data generation exactly (same RNG seed => same dataset), but
emits batched multi-row INSERT statements to a file instead of executing them over
a live connection.

Usage:
    python db/generate_seed_sql.py --out db/supabase_seed.sql
"""
import argparse
import datetime
import random
from pathlib import Path

FIRST_NAMES = [
    "James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael", "Linda",
    "David", "Elizabeth", "William", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
    "Thomas", "Sarah", "Charles", "Karen", "Wei", "Priya", "Hiro", "Fatima", "Carlos",
    "Ana", "Liam", "Olivia", "Noah", "Emma", "Yuki", "Chen", "Amara", "Diego", "Sofia",
]
LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
    "Taylor", "Thomas", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson", "White",
    "Kim", "Patel", "Nguyen", "Chen", "Silva", "Khan",
]
REGIONS = ["North America", "Europe", "APAC", "LATAM"]
SEGMENTS = ["Enterprise", "SMB", "Consumer"]
EMPLOYEE_ROLES = ["Account Executive", "Sales Manager", "SDR"]
ORDER_STATUSES = ["pending", "shipped", "delivered", "cancelled", "returned"]
ORDER_STATUS_WEIGHTS = [0.08, 0.22, 0.55, 0.08, 0.07]
CHANNELS = ["online", "retail", "phone"]
PAYMENT_METHODS = ["credit_card", "paypal", "bank_transfer", "invoice"]
PAYMENT_STATUSES = ["paid", "failed", "refunded", "pending"]
PAYMENT_STATUS_WEIGHTS = [0.85, 0.05, 0.05, 0.05]

PRODUCT_CATALOG = {
    "Electronics": [
        ("Wireless Mouse", 24.99), ("Mechanical Keyboard", 89.99), ("27in Monitor", 249.99),
        ("USB-C Hub", 39.99), ("Webcam 1080p", 54.99), ("Noise Cancelling Headphones", 179.99),
        ("Laptop Stand", 34.99), ("External SSD 1TB", 109.99), ("Bluetooth Speaker", 59.99),
        ("Docking Station", 129.99),
    ],
    "Office Supplies": [
        ("Printer Paper (Case)", 42.50), ("Stapler", 8.99), ("Sticky Notes (Pack)", 5.49),
        ("Ballpoint Pens (Box)", 12.99), ("Whiteboard 4x6", 89.00), ("Binder Clips (Box)", 6.25),
        ("Desk Organizer", 19.99), ("Label Maker", 44.99), ("Shredder", 74.99), ("File Cabinet", 149.99),
    ],
    "Furniture": [
        ("Ergonomic Office Chair", 249.00), ("Standing Desk", 399.00), ("Bookshelf", 129.00),
        ("Conference Table", 899.00), ("Guest Chair", 99.00), ("Monitor Arm", 69.00),
        ("Filing Cabinet", 159.00), ("Reception Sofa", 549.00), ("Cubicle Partition", 219.00),
        ("Desk Lamp", 29.99),
    ],
    "Software": [
        ("Analytics Suite License", 599.00), ("CRM Seat (Annual)", 480.00),
        ("Design Tools Bundle", 240.00), ("Project Mgmt Seat (Annual)", 96.00),
        ("Security Suite License", 350.00), ("Cloud Backup (Annual)", 120.00),
        ("Video Conferencing Seat", 144.00), ("Dev Tools License", 199.00),
        ("HR Platform Seat (Annual)", 210.00), ("BI Dashboard License", 720.00),
    ],
}


def esc(v) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, (datetime.date, datetime.datetime)):
        return f"'{v.isoformat()}'"
    return "'" + str(v).replace("'", "''") + "'"


def batched_insert(table: str, columns: list, rows: list, batch_size: int = 200) -> list:
    statements = []
    col_list = ", ".join(columns)
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        values = ",\n  ".join("(" + ", ".join(esc(v) for v in row) + ")" for row in chunk)
        statements.append(f"INSERT INTO {table} ({col_list}) VALUES\n  {values};")
    return statements


def daterange_random(start: datetime.date, end: datetime.date) -> datetime.date:
    delta_days = (end - start).days
    return start + datetime.timedelta(days=random.randint(0, delta_days))


def generate(n_customers: int, n_employees: int, n_orders: int) -> dict:
    random.seed(42)
    today = datetime.date(2026, 7, 4)
    three_years_ago = today - datetime.timedelta(days=3 * 365)

    customers = []
    for i in range(1, n_customers + 1):
        fn, ln = random.choice(FIRST_NAMES), random.choice(LAST_NAMES)
        email = f"{fn.lower()}.{ln.lower()}{i}@example.com"
        region = random.choice(REGIONS)
        segment = random.choices(SEGMENTS, weights=[0.2, 0.35, 0.45])[0]
        signup = daterange_random(three_years_ago, today)
        customers.append((i, f"{fn} {ln}", email, region, segment, signup))

    employees = []
    for i in range(1, n_employees + 1):
        fn, ln = random.choice(FIRST_NAMES), random.choice(LAST_NAMES)
        region = random.choice(REGIONS)
        role = random.choice(EMPLOYEE_ROLES)
        hire = daterange_random(three_years_ago, today)
        employees.append((i, f"{fn} {ln}", region, role, hire))

    products = []
    product_id_by_category = {}
    pid = 1
    for category, items in PRODUCT_CATALOG.items():
        product_id_by_category[category] = []
        for name, price in items:
            cost = round(price * random.uniform(0.4, 0.7), 2)
            products.append((pid, name, category, price, cost))
            product_id_by_category[category].append((pid, price))
            pid += 1
    all_products = [p for lst in product_id_by_category.values() for p in lst]

    orders, order_items, payments = [], [], []
    order_item_id = 1
    payment_id = 1
    customer_ids = [c[0] for c in customers]
    employee_ids = [e[0] for e in employees]

    for order_id in range(1, n_orders + 1):
        customer_id = random.choice(customer_ids)
        employee_id = random.choice(employee_ids) if random.random() > 0.1 else None
        order_date = daterange_random(three_years_ago, today)
        status = random.choices(ORDER_STATUSES, weights=ORDER_STATUS_WEIGHTS)[0]
        channel = random.choices(CHANNELS, weights=[0.6, 0.25, 0.15])[0]
        orders.append((order_id, customer_id, employee_id, order_date, status, channel))

        n_items = random.randint(1, 4)
        order_total = 0.0
        for _ in range(n_items):
            product_id, price = random.choice(all_products)
            quantity = random.randint(1, 5)
            discount = random.choices([0, 5, 10, 15], weights=[0.6, 0.2, 0.15, 0.05])[0]
            line_price = round(price * (1 - discount / 100), 2)
            order_total += line_price * quantity
            order_items.append((order_item_id, order_id, product_id, quantity, line_price, discount))
            order_item_id += 1

        if status != "pending":
            pay_status = random.choices(PAYMENT_STATUSES, weights=PAYMENT_STATUS_WEIGHTS)[0]
            method = random.choice(PAYMENT_METHODS)
            paid_at = datetime.datetime.combine(
                order_date, datetime.time(random.randint(0, 23), random.randint(0, 59))
            )
            payments.append((payment_id, order_id, round(order_total, 2), method, pay_status, paid_at))
            payment_id += 1

    return {
        "customers": customers, "employees": employees, "products": products,
        "orders": orders, "order_items": order_items, "payments": payments,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--customers", type=int, default=500)
    parser.add_argument("--employees", type=int, default=25)
    parser.add_argument("--orders", type=int, default=5000)
    parser.add_argument("--out", default=str(Path(__file__).parent / "supabase_seed.sql"))
    args = parser.parse_args()

    data = generate(args.customers, args.employees, args.orders)
    full_schema_sql = (Path(__file__).parent / "schema.sql").read_text(encoding="utf-8")
    # Drop the read-only-role section: it GRANTs on a database literally named
    # "analytics" (this project's local docker-compose DB name), which doesn't match
    # a hosted provider's database name (e.g. Supabase's is "postgres"), and app-level
    # DB privileges aren't required for this demo -- the guardrail is the real gate.
    schema_sql = full_schema_sql.split("-- Read-only analytics role")[0]

    lines = [
        "-- Auto-generated by db/generate_seed_sql.py -- schema + deterministic synthetic seed data.",
        "-- Paste this entire file into your hosted Postgres provider's SQL editor and run it.",
        "",
        "BEGIN;",
        "",
        schema_sql,
        "",
    ]
    lines += batched_insert("customers", ["customer_id", "name", "email", "region", "segment", "signup_date"], data["customers"])
    lines.append("SELECT setval('customers_customer_id_seq', (SELECT MAX(customer_id) FROM customers));")
    lines += batched_insert("employees", ["employee_id", "name", "region", "role", "hire_date"], data["employees"])
    lines.append("SELECT setval('employees_employee_id_seq', (SELECT MAX(employee_id) FROM employees));")
    lines += batched_insert("products", ["product_id", "name", "category", "unit_price", "unit_cost"], data["products"])
    lines.append("SELECT setval('products_product_id_seq', (SELECT MAX(product_id) FROM products));")
    lines += batched_insert("orders", ["order_id", "customer_id", "employee_id", "order_date", "status", "channel"], data["orders"])
    lines.append("SELECT setval('orders_order_id_seq', (SELECT MAX(order_id) FROM orders));")
    lines += batched_insert("order_items", ["order_item_id", "order_id", "product_id", "quantity", "unit_price", "discount_pct"], data["order_items"])
    lines.append("SELECT setval('order_items_order_item_id_seq', (SELECT MAX(order_item_id) FROM order_items));")
    lines += batched_insert("payments", ["payment_id", "order_id", "amount", "method", "status", "paid_at"], data["payments"])
    lines.append("SELECT setval('payments_payment_id_seq', (SELECT MAX(payment_id) FROM payments));")
    lines.append("")
    lines.append("COMMIT;")

    out_path = Path(args.out)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"Wrote {out_path} ({size_mb:.2f} MB)")
    print(f"  customers={len(data['customers'])} employees={len(data['employees'])} "
          f"products={len(data['products'])} orders={len(data['orders'])} "
          f"order_items={len(data['order_items'])} payments={len(data['payments'])}")


if __name__ == "__main__":
    main()
