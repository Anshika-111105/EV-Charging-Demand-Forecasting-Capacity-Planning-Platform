"""FastAPI application factory and lifecycle management."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ev_forecasting.api.dependencies import get_app_config, get_forecast_service
from ev_forecasting.api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ev_forecasting.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown hooks."""
    config = get_app_config()
    logger.info("Initializing EV Forecasting API in '%s' environment...", config.environment)
    # Pre-warm forecast service
    svc = get_forecast_service()
    if svc.model is not None:
        logger.info("Production model '%s' ready for serving.", svc.model.name)
    else:
        logger.warning(
            "No production model loaded at startup. Will initialize upon first train/request."
        )
    yield
    logger.info("Shutting down EV Forecasting API.")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="EV Charging Demand Forecasting & Capacity Planning API",
        description=(
            "Production-grade time-series forecasting API for multi-station EV charging demand "
            "with integrated capacity risk analytics, SQL audit logging, and provenance tracking."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routes
    app.include_router(router)
    return app


app = create_app()
