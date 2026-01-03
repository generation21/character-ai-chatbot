import json
import time

import requests


def test_chat_endpoint():
    url = "http://localhost:8000/chat"
    payload = {"message": "안녕, 프리렌. 마법에 대해 알려줘."}
    headers = {"Content-Type": "application/json"}

    print(f"Testing API at {url}...")
    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            print("✅ Success!")
            print("Response:", json.dumps(response.json(), indent=2, ensure_ascii=False))
        else:
            print(f"❌ Failed with status code {response.status_code}")
            print("Response:", response.text)
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to the server. Is it running?")


if __name__ == "__main__":
    test_chat_endpoint()
