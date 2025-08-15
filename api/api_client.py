# app/api_client.py
import os
import requests

BASE = os.getenv("OPTIM_API_URL", "").rstrip("/")

def get(path: str):
    r = requests.get(f"{BASE}{path}")
    r.raise_for_status()
    return r.json()

def post(path: str, payload: dict | None = None):
    r = requests.post(f"{BASE}{path}", json=payload or {})
    r.raise_for_status()
    return r.json()