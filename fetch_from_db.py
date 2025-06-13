from sqlalchemy import create_engine
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Your database URI
uri = "postgresql://sy91dhb:g5t49ao@178.156.142.230:5432/hd64m1ki"
engine = create_engine(uri)

import math
from sqlalchemy import text

# Configuration
TOTAL_ESTIMATED_BYTES = 6754 * 1024 * 1024  # 6754 MB
TARGET_FILES = 7
BYTES_PER_FILE = math.ceil(TOTAL_ESTIMATED_BYTES / TARGET_FILES)
CHUNK_SIZE = 50_000  # Rows per batch

# SQL Query
query = text("""
    SELECT id, text 
    FROM hacker_news.items
    WHERE type = 'comment' 
      AND text IS NOT NULL
      AND kids IS NOT NULL
    ORDER BY id
""")

# Initialize
file_index = 1
current_bytes = 0
comments_written = 0
file = open(f'hn_comments_{file_index}.txt', 'w', encoding='utf-8')

try:
    with engine.connect() as conn:
        result = conn.execution_options(stream_results=True, yield_per=CHUNK_SIZE).execute(query)
        
        for row in result:
            comment_text = f"<COMMENT>{row.id}\n{row.text}\n"  # ID as metadata
            comment_bytes = len(comment_text.encode('utf-8'))
            
            # Switch to new file if needed
            if current_bytes + comment_bytes > BYTES_PER_FILE and file_index < TARGET_FILES:
                file.close()
                file_index += 1
                current_bytes = 0
                file = open(f'hn_comments_{file_index}.txt', 'w', encoding='utf-8')
            
            file.write(comment_text)
            current_bytes += comment_bytes
            comments_written += 1
            
            # Progress tracking
            if comments_written % 10000 == 0:
                print(f"Written {comments_written:,} comments | File {file_index}/{TARGET_FILES} | {current_bytes/1024/1024:.1f}MB")
finally:
    file.close()

print(f"Done! Split {comments_written:,} comments into {file_index} files")