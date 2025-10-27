import sqlite3
import pandas as pd
import os

# === Config ===
CSV_FILE = "weather_data.csv"
DB_FILE = "weather_data.db"
TABLE_NAME = "weather_data"

# === 1. Load CSV ===
if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(f"CSV file not found: {CSV_FILE}")

df = pd.read_csv(CSV_FILE)
print(f"✅ Loaded {len(df)} rows from {CSV_FILE}")

# === 2. Connect to SQLite ===
conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()

# === 3. Create table if not exists ===
# Automatically infer schema from CSV columns
columns = ", ".join([f"{col} TEXT" for col in df.columns])
cursor.execute(f"CREATE TABLE IF NOT EXISTS {TABLE_NAME} ({columns});")

# === 4. Clear existing data ===
cursor.execute(f"DELETE FROM {TABLE_NAME};")
print("🧹 Old data cleared from database")

# === 5. Insert fresh data ===
df.to_sql(TABLE_NAME, conn, if_exists='append', index=False)
print(f"💾 Inserted {len(df)} rows into {DB_FILE}")

# === 6. Close ===
conn.commit()
conn.close()
print("✅ Sync complete: database now matches CSV")