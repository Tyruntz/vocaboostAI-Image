#!/bin/bash
echo "🚀 Installing requirements..."
pip install -r requirements.txt

echo "🔥 Starting FastAPI..."
uvicorn app:app --host 0.0.0.0 --port 8000
