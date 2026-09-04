"""Convenience launcher: `python services/api/run.py` from the project root."""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False, app_dir="services/api")
