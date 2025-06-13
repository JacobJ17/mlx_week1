from dotenv import load_dotenv
import os
import psycopg2
import json

load_dotenv()

def get_connection():
  return psycopg2.connect(
    database=os.getenv("DB_NAME"),
    host=os.getenv("DB_HOST"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    port=os.getenv("DB_PORT")
  )

def log(data):
  row_id = None
  try:
    db = get_connection()
    with db.cursor() as cursor:
      # Prepare data for items_logs table
      columns = ['timestamp', 'prediction', 'item']
      placeholders = ', '.join(['%s'] * len(columns))
      values = [
        data['timestamp'],
        data['prediction'],
        json.dumps(data['item']) if not isinstance(data['item'], str) else data['item']
      ]
      sql = f"""
        INSERT INTO items_logs ({', '.join(columns)})
        VALUES ({placeholders})
        RETURNING id;
      """
      cursor.execute(sql, values)
      row = cursor.fetchone()
      if row:
        row_id = row[0]
    db.commit()
    db.close()
  except (Exception, psycopg2.DatabaseError) as error:
    print("Error inserting log:", error)
  return row_id

def fetch_logs(limit=100):
  logs = []
  try:
    db = get_connection()
    with db.cursor() as cursor:
      cursor.execute(
        '''
        SELECT *
        FROM items_logs
        ORDER BY timestamp DESC
        LIMIT %s;
        ''',
        (limit,)
      )
      logs = cursor.fetchall()
    db.commit()
    db.close()
  except (Exception, psycopg2.DatabaseError) as error:
    print("Error fetching logs:", error)
  return logs
