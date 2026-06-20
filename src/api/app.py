import uvicorn
from fastapi import FastAPI
from src.api.middleware import setup_middleware
from src.api.routes import health, analyze, stream
from src.utils.logger import setup_logging, get_logger

setup_logging()
logger = get_logger("api_app")

# Initialize app
app = FastAPI(
    title="AI Micro-Expression & Stress Analyzer API",
    description="REST API and WebSocket stream server for real-time facial micro-expression features and stress estimation.",
    version="0.1.0"
)

# Setup middleware
setup_middleware(app)

# Include routes
app.include_router(health.router)
app.include_router(analyze.router)
app.include_router(stream.router)

@app.get("/")
async def root():
    return {
        "message": "Welcome to the AI Micro-Expression & Stress Analyzer API",
        "docs": "/docs",
        "health": "/health"
    }

def main():
    logger.info("Starting API Server...")
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=False)

if __name__ == "__main__":
    main()
