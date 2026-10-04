import sqlite3
import time
import functools
import json
import os

DB_NAME = "rag_trace_logs.db"


def init_trace_db():
    """Trace logları için SQLite veritabanını ve tabloyu oluşturur."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trace_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            step_name TEXT,
            function_name TEXT,
            execution_time_ms REAL,
            inputs TEXT,
            outputs TEXT,
            status TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()


def trace_step(step_name):
    """
    Fonksiyonların çalışma süresini, girdilerini ve çıktılarını yakalayarak
    veritabanına kaydeden dekoratör.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            status = "SUCCESS"
            output_data = None

            try:
                result = func(*args, **kwargs)
                output_data = result
                return result
            except Exception as e:
                status = f"ERROR: {str(e)}"
                output_data = status
                raise e
            finally:
                end_time = time.time()
                execution_time_ms = (end_time - start_time) * 1000

                # Sadece keyword argümanları (kwargs) loglayarak büyük objeleri filtreliyoruz
                safe_inputs = {k: str(v)[:200] for k, v in kwargs.items()}

                # Çıktı çok büyükse (örneğin tüm PDF metni) özetle
                safe_output = str(output_data)[:500] + "..." if len(str(output_data)) > 500 else str(output_data)

                # Veritabanına yaz
                conn = sqlite3.connect(DB_NAME)
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO trace_logs (step_name, function_name, execution_time_ms, inputs, outputs, status)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (step_name, func.__name__, execution_time_ms, json.dumps(safe_inputs), safe_output, status))
                conn.commit()
                conn.close()

        return wrapper

    return decorator


# Veritabanını başlat
init_trace_db()