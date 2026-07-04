-- Enterprise Text-to-SQL Analytics Agent -- sample sales/orders schema
-- A synthetic but realistic OLTP-ish schema used as the target for NL->SQL generation.

DROP TABLE IF EXISTS payments CASCADE;
DROP TABLE IF EXISTS order_items CASCADE;
DROP TABLE IF EXISTS orders CASCADE;
DROP TABLE IF EXISTS products CASCADE;
DROP TABLE IF EXISTS employees CASCADE;
DROP TABLE IF EXISTS customers CASCADE;

CREATE TABLE customers (
    customer_id     SERIAL PRIMARY KEY,
    name            TEXT NOT NULL,
    email           TEXT NOT NULL UNIQUE,
    region          TEXT NOT NULL,           -- 'North America', 'Europe', 'APAC', 'LATAM'
    segment         TEXT NOT NULL,           -- 'Enterprise', 'SMB', 'Consumer'
    signup_date     DATE NOT NULL
);

CREATE TABLE employees (
    employee_id     SERIAL PRIMARY KEY,
    name            TEXT NOT NULL,
    region          TEXT NOT NULL,
    role            TEXT NOT NULL,           -- 'Account Executive', 'Sales Manager', 'SDR'
    hire_date       DATE NOT NULL
);

CREATE TABLE products (
    product_id      SERIAL PRIMARY KEY,
    name            TEXT NOT NULL,
    category        TEXT NOT NULL,           -- 'Electronics', 'Office Supplies', 'Furniture', 'Software'
    unit_price      NUMERIC(10, 2) NOT NULL,
    unit_cost       NUMERIC(10, 2) NOT NULL
);

CREATE TABLE orders (
    order_id        SERIAL PRIMARY KEY,
    customer_id     INTEGER NOT NULL REFERENCES customers(customer_id),
    employee_id     INTEGER REFERENCES employees(employee_id),
    order_date      DATE NOT NULL,
    status          TEXT NOT NULL,           -- 'pending', 'shipped', 'delivered', 'cancelled', 'returned'
    channel         TEXT NOT NULL            -- 'online', 'retail', 'phone'
);

CREATE TABLE order_items (
    order_item_id   SERIAL PRIMARY KEY,
    order_id        INTEGER NOT NULL REFERENCES orders(order_id),
    product_id      INTEGER NOT NULL REFERENCES products(product_id),
    quantity        INTEGER NOT NULL,
    unit_price      NUMERIC(10, 2) NOT NULL, -- price at time of sale (may differ from products.unit_price)
    discount_pct    NUMERIC(5, 2) NOT NULL DEFAULT 0
);

CREATE TABLE payments (
    payment_id      SERIAL PRIMARY KEY,
    order_id        INTEGER NOT NULL REFERENCES orders(order_id),
    amount          NUMERIC(10, 2) NOT NULL,
    method          TEXT NOT NULL,           -- 'credit_card', 'paypal', 'bank_transfer', 'invoice'
    status          TEXT NOT NULL,           -- 'paid', 'failed', 'refunded', 'pending'
    paid_at         TIMESTAMP
);

CREATE INDEX idx_orders_customer_id ON orders(customer_id);
CREATE INDEX idx_orders_employee_id ON orders(employee_id);
CREATE INDEX idx_orders_order_date ON orders(order_date);
CREATE INDEX idx_order_items_order_id ON order_items(order_id);
CREATE INDEX idx_order_items_product_id ON order_items(product_id);
CREATE INDEX idx_payments_order_id ON payments(order_id);

-- Read-only analytics role used by the app's connection pool.
-- The application DB user should NOT have DDL/DML privileges beyond SELECT;
-- the guardrail layer is defense-in-depth, not the only line of defense.
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'sql_agent_readonly') THEN
        CREATE ROLE sql_agent_readonly LOGIN PASSWORD 'readonly_pw';
    END IF;
END
$$;

GRANT CONNECT ON DATABASE analytics TO sql_agent_readonly;
GRANT USAGE ON SCHEMA public TO sql_agent_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO sql_agent_readonly;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO sql_agent_readonly;
