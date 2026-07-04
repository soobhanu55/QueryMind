"""
Generates synthetic sales/orders data for the sample analytics schema (db/schema.sql).

Deterministic (fixed RNG seed) so accuracy/load benchmarks are reproducible across runs.

Usage:
    python db/seed.py                 # uses DATABASE_URL from env / .env
    python db/seed.py --customers 500 --orders 5000
"""
import argparse
import asyncio
import datetime
import os
import random
import sys

import asyncpg
from dotenv import load_dotenv

load_dotenv()

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


def daterange_random(start: datetime.date, end: datetime.date) -> datetime.date:
    delta_days = (end - start).days
    return start + datetime.timedelta(days=random.randint(0, delta_days))


async def seed(conn: asyncpg.Connection, n_customers: int, n_employees: int, n_orders: int):
    random.seed(42)
    today = datetime.date(2026, 7, 4)
    three_years_ago = today - datetime.timedelta(days=3 * 365)

    print(f"Seeding {n_customers} customers...")
    customer_ids = []
    for i in range(n_customers):
        fn, ln = random.choice(FIRST_NAMES), random.choice(LAST_NAMES)
        email = f"{fn.lower()}.{ln.lower()}{i}@example.com"
        region = random.choice(REGIONS)
        segment = random.choices(SEGMENTS, weights=[0.2, 0.35, 0.45])[0]
        signup = daterange_random(three_years_ago, today)
        cid = await conn.fetchval(
            "INSERT INTO customers (name, email, region, segment, signup_date) "
            "VALUES ($1,$2,$3,$4,$5) RETURNING customer_id",
            f"{fn} {ln}", email, region, segment, signup,
        )
        customer_ids.append(cid)

    print(f"Seeding {n_employees} employees...")
    employee_ids = []
    for i in range(n_employees):
        fn, ln = random.choice(FIRST_NAMES), random.choice(LAST_NAMES)
        region = random.choice(REGIONS)
        role = random.choice(EMPLOYEE_ROLES)
        hire = daterange_random(three_years_ago, today)
        eid = await conn.fetchval(
            "INSERT INTO employees (name, region, role, hire_date) VALUES ($1,$2,$3,$4) "
            "RETURNING employee_id",
            f"{fn} {ln}", region, role, hire,
        )
        employee_ids.append(eid)

    print("Seeding products...")
    product_ids = {}  # category -> list of product_id
    for category, items in PRODUCT_CATALOG.items():
        product_ids[category] = []
        for name, price in items:
            cost = round(price * random.uniform(0.4, 0.7), 2)
            pid = await conn.fetchval(
                "INSERT INTO products (name, category, unit_price, unit_cost) "
                "VALUES ($1,$2,$3,$4) RETURNING product_id",
                name, category, price, cost,
            )
            product_ids[category].append((pid, price))

    all_products = [p for lst in product_ids.values() for p in lst]

    print(f"Seeding {n_orders} orders (with line items + payments)...")
    for i in range(n_orders):
        customer_id = random.choice(customer_ids)
        employee_id = random.choice(employee_ids) if random.random() > 0.1 else None
        order_date = daterange_random(three_years_ago, today)
        status = random.choices(ORDER_STATUSES, weights=ORDER_STATUS_WEIGHTS)[0]
        channel = random.choices(CHANNELS, weights=[0.6, 0.25, 0.15])[0]

        order_id = await conn.fetchval(
            "INSERT INTO orders (customer_id, employee_id, order_date, status, channel) "
            "VALUES ($1,$2,$3,$4,$5) RETURNING order_id",
            customer_id, employee_id, order_date, status, channel,
        )

        n_items = random.randint(1, 4)
        order_total = 0.0
        for _ in range(n_items):
            product_id, price = random.choice(all_products)
            quantity = random.randint(1, 5)
            discount = random.choices([0, 5, 10, 15], weights=[0.6, 0.2, 0.15, 0.05])[0]
            line_price = round(price * (1 - discount / 100), 2)
            order_total += line_price * quantity
            await conn.execute(
                "INSERT INTO order_items (order_id, product_id, quantity, unit_price, discount_pct) "
                "VALUES ($1,$2,$3,$4,$5)",
                order_id, product_id, quantity, line_price, discount,
            )

        if status != "pending":
            pay_status = random.choices(PAYMENT_STATUSES, weights=PAYMENT_STATUS_WEIGHTS)[0]
            method = random.choice(PAYMENT_METHODS)
            paid_at = datetime.datetime.combine(order_date, datetime.time(random.randint(0, 23), random.randint(0, 59)))
            await conn.execute(
                "INSERT INTO payments (order_id, amount, method, status, paid_at) VALUES ($1,$2,$3,$4,$5)",
                order_id, round(order_total, 2), method, pay_status, paid_at,
            )

        if (i + 1) % 1000 == 0:
            print(f"  ...{i + 1}/{n_orders} orders")

    print("Done seeding.")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--customers", type=int, default=500)
    parser.add_argument("--employees", type=int, default=25)
    parser.add_argument("--orders", type=int, default=5000)
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL_SYNC"))
    args = parser.parse_args()

    dsn = args.database_url or os.getenv("DATABASE_URL_SYNC") or \
        "postgresql://postgres:postgres@localhost:5432/analytics"
    # asyncpg needs a plain postgres:// dsn, not the sqlalchemy +asyncpg driver string
    dsn = dsn.replace("postgresql+asyncpg://", "postgresql://")

    conn = await asyncpg.connect(dsn)
    try:
        with open(os.path.join(os.path.dirname(__file__), "schema.sql")) as f:
            schema_sql = f.read()
        print("Applying schema.sql ...")
        await conn.execute(schema_sql)

        row_count = await conn.fetchval("SELECT count(*) FROM customers")
        if row_count and row_count > 0:
            print("Tables already contain data; truncating before reseeding.")
            await conn.execute(
                "TRUNCATE payments, order_items, orders, products, employees, customers RESTART IDENTITY CASCADE"
            )

        await seed(conn, args.customers, args.employees, args.orders)
    finally:
        await conn.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()) or 0)
