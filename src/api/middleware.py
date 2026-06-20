from fastapi.middleware.cors import CORSMiddleware

def setup_middleware(app):
    """
    Sets up middleware on the FastAPI application, including CORS config.
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
