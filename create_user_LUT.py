import math
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sqlalchemy import create_engine, text
from tqdm import tqdm
import pickle

# Your database URI
uri = "postgresql://sy91dhb:g5t49ao@178.156.142.230:5432/hd64m1ki"
engine = create_engine(uri)

def create_user_lookup(engine):
    """Create a username lookup dictionary with datetime conversion and NULL filtering"""
    user_lookup = {}
    null_id_count = 0
    
    # SQL query with explicit NULL check for id
    query = text("""
        SELECT 
            id as username,
            created,
            karma,
            submitted
        FROM hacker_news.users
        WHERE id IS NOT NULL
    """)
    
    with engine.connect() as conn:
        # Get total rows for progress bar (excluding NULL ids)
        total_rows = conn.execute(
            text("SELECT COUNT(*) FROM hacker_news.users WHERE id IS NOT NULL")
        ).scalar()
        
        # Count how many NULL ids we're excluding
        null_id_count = conn.execute(
            text("SELECT COUNT(*) FROM hacker_news.users WHERE id IS NULL")
        ).scalar()
        
        if null_id_count > 0:
            print(f"Excluding {null_id_count} rows with NULL user IDs")
        
        # Execute main query with progress bar
        results = conn.execute(query)
        
        for row in tqdm(results, total=total_rows, desc="Building user lookup"):
            # Convert PostgreSQL timestamp to Python datetime
            created_dt = row.created if isinstance(row.created, datetime) else datetime.strptime(row.created, '%Y-%m-%d %H:%M:%S.%f')
            
            # Handle submitted field - now assuming it's already a list
            submitted_ids = row.submitted if row.submitted is not None else []
            
            user_lookup[row.username] = {
                'created': created_dt,
                'karma': row.karma if row.karma is not None else 0,
                'submitted_count': len(submitted_ids)
            }
    
    print(f"Successfully loaded {len(user_lookup)} users")
    return user_lookup

# Create the lookup dictionary
user_lookup = create_user_lookup(engine)
# Save the dictionary
with open('user_lookup.pkl', 'wb') as f:
    pickle.dump(user_lookup, f)

print("Saved user_lookup.pkl")

# Example usage:
sample_user = next(iter(user_lookup.items()))
print("\nSample user entry:")
print(f"Username: {sample_user[0]}")
print(f"Created: {sample_user[1]['created']} (type: {type(sample_user[1]['created'])})")
print(f"Karma: {sample_user[1]['karma']}")
print(f"Submitted count: {sample_user[1]['submitted_count']}")