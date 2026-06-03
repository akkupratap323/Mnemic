from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from graph_service.config import get_settings
from graph_service.routers import ingest, retrieve
from graph_service.mnemic_graph import initialize_mnemic


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    await initialize_mnemic(settings)
    yield
    # Shutdown
    # No need to close Mnemic here, as it's handled per-request


app = FastAPI(lifespan=lifespan)


app.include_router(retrieve.router)
app.include_router(ingest.router)


@app.get('/healthcheck')
async def healthcheck():
    return JSONResponse(content={'status': 'healthy'}, status_code=200)
