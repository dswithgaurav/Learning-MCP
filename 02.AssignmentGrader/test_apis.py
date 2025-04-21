import requests
import os
from dotenv import load_dotenv

load_dotenv()

url = "https://www.googleapis.com/customsearch/v1"

params = {
    "key": os.getenv("SEARCH_API"),
    "cx": os.getenv("SEARCH_INDEX"),
    "q": "Model Context Protocol",
}
print(params)
resp = requests.get(url=url, params=params)
print(resp.json())
