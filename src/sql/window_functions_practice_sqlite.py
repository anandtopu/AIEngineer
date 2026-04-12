from __future__ import annotations

import sqlite3


def run(conn: sqlite3.Connection, sql: str):
    cur = conn.cursor()
    cur.execute(sql)
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description] if cur.description else []
    return cols, rows


def print_rows(title: str, cols: list[str], rows: list[tuple]):
    print(f"\n== {title} ==")
    print(" | ".join(cols))
    for r in rows:
        print(" | ".join(str(x) for x in r))


def main():
    # SQLite supports window functions (ROW_NUMBER, LAG, SUM OVER, etc.).
    conn = sqlite3.connect(":memory:")

    conn.executescript(
        """
        CREATE TABLE orders (
            user_id TEXT NOT NULL,
            order_id TEXT NOT NULL,
            order_date TEXT NOT NULL,
            revenue REAL NOT NULL
        );

        INSERT INTO orders(user_id, order_id, order_date, revenue) VALUES
            ('u1', 'o1', '2026-01-01', 10.0),
            ('u1', 'o2', '2026-01-10', 25.0),
            ('u1', 'o3', '2026-02-01', 15.0),
            ('u2', 'o4', '2026-01-03', 7.0),
            ('u2', 'o5', '2026-01-20', 9.0),
            ('u3', 'o6', '2026-01-02', 100.0);
        """
    )

    # 1) Rank each user's orders by date.
    cols, rows = run(
        conn,
        """
        SELECT
            user_id,
            order_id,
            order_date,
            revenue,
            ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY order_date) AS order_number
        FROM orders
        ORDER BY user_id, order_date;
        """,
    )
    print_rows("ROW_NUMBER per user", cols, rows)

    # 2) First purchase date per user (window function variant).
    cols, rows = run(
        conn,
        """
        SELECT
            user_id,
            order_id,
            order_date,
            MIN(order_date) OVER (PARTITION BY user_id) AS first_order_date
        FROM orders
        ORDER BY user_id, order_date;
        """,
    )
    print_rows("First order date per user", cols, rows)

    # 3) Running revenue per user.
    cols, rows = run(
        conn,
        """
        SELECT
            user_id,
            order_id,
            order_date,
            revenue,
            SUM(revenue) OVER (
                PARTITION BY user_id
                ORDER BY order_date
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS running_revenue
        FROM orders
        ORDER BY user_id, order_date;
        """,
    )
    print_rows("Running revenue per user", cols, rows)

    # 4) Days since previous order (LAG).
    cols, rows = run(
        conn,
        """
        SELECT
            user_id,
            order_id,
            order_date,
            LAG(order_date) OVER (PARTITION BY user_id ORDER BY order_date) AS prev_order_date,
            CAST(
                julianday(order_date) - julianday(LAG(order_date) OVER (PARTITION BY user_id ORDER BY order_date))
                AS INT
            ) AS days_since_prev
        FROM orders
        ORDER BY user_id, order_date;
        """,
    )
    print_rows("Days since previous order", cols, rows)


if __name__ == "__main__":
    main()
