import db
from datetime import datetime
import streamlit as st
from get_hn_item import process_hn_url as fetch
import re

default_session_values = [st.session_state.setdefault(k, v) for k, v in {
    'url': None,
    'predicted_score': '--',
    'feedback': '',
    'item': None,
    'logs': db.fetch_logs(),
}.items()]

def is_valid_hackernews_url(url):
    # Hacker News item URL patterns:
    # https://news.ycombinator.com/item?id=<id>
    # http://news.ycombinator.com/item?id=<id>
    # //news.ycombinator.com/item?id=<id>
    # news.ycombinator.com/item?id=<id>

    pattern = r'^(https?:\/\/)?(www\.)?news\.ycombinator\.com\/item\?id=\d+$'
    return re.match(pattern, url) is not None

def get_prediction(url):
    st.session_state.item = fetch(url)
    return 42

def predict_score_btn():
    st.session_state.feedback = ""

    if (
        st.session_state.url == None or
        not is_valid_hackernews_url(st.session_state.url)
    ):
        st.session_state.feedback = "Please enter a HackerNews URL"
        return

    st.session_state.predicted_score = get_prediction(st.session_state.url)
    db.log({
      'timestamp': datetime.now(),
      'item': st.session_state.item,
      'prediction': st.session_state.predicted_score,
    })
    st.session_state.logs = db.fetch_logs()
    st.session_state.feedback = ""

# Web App
st.write('# HackerNews Score Prediction')
url = st.text_input(
    'URL',
    placeholder='Enter a valid hacker news item url',
    value=None,
    key="url"
)
button = st.button("Predict Score!", on_click=predict_score_btn)
st.write(st.session_state.feedback)
if(st.session_state.item):
    st.write('## Hacker News item', st.session_state.item)
st.write('## Predicted Score:', st.session_state.predicted_score)

logs_table = """
---
## History
| Timestamp | Predicted Score | Item |
|-----------|-----------------|------|
"""
for row in st.session_state.logs:
    _, timestamp, prediction, item = row
    timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    logs_table += f"| {timestamp} | {prediction} | {item} |\n"

st.markdown(logs_table)
