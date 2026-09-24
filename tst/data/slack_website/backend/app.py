"""
FastAPI application entry point for Slack Workspace Manager.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from db_extensions import ExtendedDatabase
from api.routes import router, set_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager - handles startup and shutdown."""
    # Startup: Initialize database with empty schema
    db = ExtendedDatabase(connection_string=settings.connection_string)

    # Set database in routes module
    set_db(db)

    # Store in app state for access
    app.state.db = db

    print("Slack Workspace Manager API started")

    yield

    # Shutdown: Cleanup if needed
    print("Slack Workspace Manager API shutting down")


app = FastAPI(
    title="Slack Workspace Manager API",
    description="REST API for the Slack Workspace Manager web GUI",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api")


# Root endpoint for basic info
@app.get("/")
def root():
    """Root endpoint with API info."""
    return {
        "name": "Slack Workspace Manager API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=settings.port)
