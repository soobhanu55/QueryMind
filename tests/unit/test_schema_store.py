from app.schema_store import get_schema_store


def test_loads_all_sample_schema_tables():
    store = get_schema_store()
    assert set(store.table_names()) == {
        "customers", "employees", "products", "orders", "order_items", "payments",
    }


def test_retrieves_relevant_tables_for_revenue_question():
    store = get_schema_store()
    tables = store.retrieve_relevant_tables("What is the total revenue from order items by customer region?")
    assert "order_items" in tables
    assert "customers" in tables


def test_retrieve_relevant_tables_falls_back_when_no_keywords_match():
    store = get_schema_store()
    tables = store.retrieve_relevant_tables("asdf qwerty zzz")
    assert tables  # never returns empty -- falls back to a sane default


def test_render_schema_text_only_includes_requested_tables():
    store = get_schema_store()
    text = store.render_schema_text(["products"])
    assert "TABLE products" in text
    assert "TABLE customers" not in text
