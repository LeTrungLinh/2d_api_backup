# creat api using fastapi
from fastapi import FastAPI, Request
from services import processing
import uvicorn
import jwt
from datetime import datetime, timedelta
from fastapi.middleware.trustedhost import TrustedHostMiddleware

app = FastAPI()
app.include_router(processing.router)
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["0.0.0.0", "103.191.146.36", "*"]
)


@app.get("/")
async def root():
    return {"message": "API is working .... "}


