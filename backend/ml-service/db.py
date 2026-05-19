import os
from contextlib import contextmanager

import psycopg2
import psycopg2.extras


def db_config():
    return {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", "5432")),
        "dbname": os.getenv("DB_NAME", "mnist_database"),
        "user": os.getenv("DB_USER", "mnist_user"),
        "password": os.getenv("DB_PASSWORD", "mnist_password"),
    }


@contextmanager
def get_conn():
    conn = psycopg2.connect(**db_config())
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def current_dataset_version() -> int:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT version FROM dataset_state WHERE id = 1")
            row = cur.fetchone()
            return int(row[0])


def fetch_accepted_samples():
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("""
                SELECT id, pixels, label
                FROM digit_samples
                WHERE status = 'accepted' AND deleted = FALSE
                ORDER BY id ASC
            """)
            return cur.fetchall()
