from fastapi import FastAPI

app = FastAPI(title="3dcamera API")


@app.get("/health")
def health():
    return {"status": "ok"}
