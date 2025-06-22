import sqlite3

conn = sqlite3.connect("tracking.db")
cursor = conn.cursor()

cursor.execute("SELECT * FROM tracking_log LIMIT 10")
rows = cursor.fetchall()

for row in rows:
    print(row)

conn.close()
