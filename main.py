from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from api.routes import router as weka_router


app = FastAPI(title="WekaDemo Python API", version="2.0.0")
app.include_router(weka_router)
app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
