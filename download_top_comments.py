import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sqlalchemy import create_engine, text
from tqdm import tqdm

# Your database URI
uri = "postgresql://sy91dhb:g5t49ao@178.156.142.230:5432/hd64m1ki"
engine = create_engine(uri)

# Configuration
CHUNK_SIZE = 50_000  # Rows per batch
DELIMITER = "<COMMENT>"
REPLY_THRESHOLD = 5  # Minimum replies to consider

# SQL Query (PostgreSQL version)
query = text(f"""
    SELECT id, text 
    FROM hacker_news.items
    WHERE type = 'comment' 
      AND text IS NOT NULL
      AND ARRAY_LENGTH(kids, 1) > {REPLY_THRESHOLD}
    ORDER BY id
""")

with (
    engine.connect() as conn,
    open(f'hn_comments_gt{REPLY_THRESHOLD}_replies.txt', 'w', encoding='utf-8') as f
):
    # Get total count for progress bar
    count_query = text(f"""
        SELECT COUNT(*) 
        FROM hacker_news.items
        WHERE type = 'comment' 
          AND text IS NOT NULL
          AND ARRAY_LENGTH(kids, 1) > {REPLY_THRESHOLD}
    """)
    total_comments = conn.execute(count_query).scalar()
    
    # Stream results with progress
    result = conn.execution_options(stream_results=True, yield_per=CHUNK_SIZE).execute(query)
    
    with tqdm(total=total_comments, unit='comments') as pbar:
        first = True
        for row in result:
            if not first:
                f.write(DELIMITER)
            f.write(f"{row.id}\n{row.text}\n")  # ID on first line, text on second
            first = False
            pbar.update(1)

print(f"Done! Saved {total_comments:,} comments with >{REPLY_THRESHOLD} replies")