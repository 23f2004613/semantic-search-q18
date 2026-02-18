import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("AIPIPE_TOKEN")
openai.base_url = "https://aipipe.org/openai/v1"

try:
    resp = openai.Embedding.create(model="text-embedding-3-small", input="test")
    print("✅ Token works! Dim:", len(resp['data'][0]['embedding']))
except Exception as e:
    print("❌ Error:", e)
