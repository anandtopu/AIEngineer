# Module 07 — SQL for AI Engineers

## Goals

- Be comfortable with joins, grouping, window functions (conceptually)
- Translate product questions into SQL

## Topics

- `SELECT`, `WHERE`, `GROUP BY`, `HAVING`
- `JOIN` types
- `CASE WHEN`
- Window functions (common ones: `ROW_NUMBER`, rolling aggregates)

## Practice

- Run `src/sql/sql_practice_sqlite.py`
- Run `src/sql/window_functions_practice_sqlite.py`
- Modify queries to answer:
  - daily active users
  - retention (cohort) (exercise)

Additional window-function exercises:

- Compute each user's first purchase date and tag each order as `is_first_purchase`.
- Compute 7-day rolling revenue per user.
- Compute days since previous event/purchase.
