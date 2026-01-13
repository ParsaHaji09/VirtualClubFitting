# Commands to get the project running (local development)

How to start up the backend
```bash
source venv/bin/activate
pip install -r requirements.txt
python3 train_model.py
uvicorn main:app --reload --port 8000
```

How to start up the frontend
```bash
npm install
npm run dev
```