from fastapi import APIRouter

router = APIRouter()

@router.get("/health", tags=["health"])
async def health_check():
    """
    Standard health check endpoint.
    """
    return {"status": "healthy"}
