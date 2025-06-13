import html
import re
import requests
from urllib.parse import urlparse

def extract_item_id(url):
    parsed = urlparse(url)

    # Case 1: Standard story URL (https://news.ycombinator.com/item?id=12345)
    if 'item?id=' in url:
        return url.split('item?id=')[1].split('&')[0]

    # Case 2: New format (https://news.ycombinator.com/item/12345)
    match = re.search(r'/item/(\d+)', parsed.path)
    if match:
        return match.group(1)

    raise ValueError("Could not extract item ID from URL")

def get_hn_item(item_id):
    api_url = f"https://hacker-news.firebaseio.com/v0/item/{item_id}.json"
    response = requests.get(api_url)
    response.raise_for_status()
    return response.json()

def extract_fields(item_data):

    print(item_data)

    result = {
        'title': item_data.get('title', ''),
        'url': item_data.get('url', ''),
        'user': item_data.get('by', ''),
        'text': re.sub(r'<[^>]+>', '', html.unescape(item_data.get('text', ''))),
        'time': item_data.get('time', 0)
    }

    return result

def process_hn_url(url):
    try:
        item_id = extract_item_id(url)
        item_data = get_hn_item(item_id)
        extracted_fields = extract_fields(item_data)
        return extracted_fields
    except Exception as e:
        return {'error': str(e)}

# Test
if __name__ == "__main__":

    test_urls = [
        'https://news.ycombinator.com/item?id=44245729',
        'https://news.ycombinator.com/item?id=44245722'
    ]

    url = test_urls[0]

    result = process_hn_url(url)

    if 'error' in result:
        print(f"Error: {result['error']}")

    print(result)
    print()
