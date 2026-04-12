from __future__ import annotations

import sqlite3


def run_query(conn: sqlite3.Connection, sql: str):
    cur = conn.cursor()
    cur.execute(sql)
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description] if cur.description else []
    return cols, rows


def print_table(title: str, cols: list[str], rows: list[tuple]):
    print(f"\n== {title} ==")
    print(" | ".join(cols))
    for r in rows:
        print(" | ".join(str(x) for x in r))


def main():
    conn = sqlite3.connect(":memory:")

    conn.executescript(
        """
        CREATE TABLE events (
            user_id TEXT NOT NULL,
            event_name TEXT NOT NULL,
            event_date TEXT NOT NULL
        );

        INSERT INTO events(user_id, event_name, event_date) VALUES
            ('u1', 'open_app', '2026-01-01'),
            ('u1', 'purchase', '2026-01-01'),
            ('u2', 'open_app', '2026-01-01'),
            ('u2', 'open_app', '2026-01-02'),
            ('u3', 'open_app', '2026-01-02'),
            ('u3', 'open_app', '2026-01-03'),
            ('u4', 'open_app', '2026-01-03');
        """
    )

    # 1) Daily active users (DAU)
    cols, rows = run_query(
        conn,
        """
        SELECT
            event_date,
            COUNT(DISTINCT user_id) AS dau
        FROM events
        WHERE event_name = 'open_app'
        GROUP BY event_date
        ORDER BY event_date;
        """,
    )
    print_table("Daily Active Users", cols, rows)

    # 2) Users with at least 2 active days
    cols, rows = run_query(
        conn,
        """
        SELECT
            user_id,
            COUNT(DISTINCT event_date) AS active_days
        FROM events
        WHERE event_name = 'open_app'
        GROUP BY user_id
        HAVING COUNT(DISTINCT event_date) >= 2
        ORDER BY active_days DESC, user_id;
        """,
    )
    print_table("Users with >=2 active days", cols, rows)

    # 3) Simple next-day retention: of users active on a day, how many were also active next day
    cols, rows = run_query(
        conn,
        """
        WITH daily_users AS (
            SELECT DISTINCT user_id, event_date
            FROM events
            WHERE event_name = 'open_app'
        ),
        pairs AS (
            SELECT
                d1.event_date AS cohort_date,
                d1.user_id AS user_id,
                CASE WHEN d2.user_id IS NOT NULL THEN 1 ELSE 0 END AS retained_next_day
            FROM daily_users d1
            LEFT JOIN daily_users d2
                ON d2.user_id = d1.user_id
               AND d2.event_date = date(d1.event_date, '+1 day')
        )
        SELECT
            cohort_date,
            COUNT(*) AS cohort_size,
            SUM(retained_next_day) AS retained_users,
            ROUND(1.0 * SUM(retained_next_day) / COUNT(*), 3) AS retention_rate
        FROM pairs
        GROUP BY cohort_date
        ORDER BY cohort_date;
        """,
    )
    print_table("Next-day retention", cols, rows)


if __name__ == "__main__":
    main()
