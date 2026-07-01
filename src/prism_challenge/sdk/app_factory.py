from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Coroutine, Sequence
from contextlib import asynccontextmanager
from time import time
from typing import Any, Protocol

from fastapi import APIRouter, Depends, FastAPI

from .auth import build_internal_auth_dependency
from .config import ChallengeSettings
from .schemas import HealthResponse, VersionResponse, WeightsResponse

GetWeightsFn = Callable[[], Awaitable[dict[str, float]]]
BackgroundTaskFactory = Callable[[FastAPI], Coroutine[Any, Any, None]]

_logger = logging.getLogger("prism.sdk.app_factory")


class ChallengeDatabase(Protocol):
    async def init(self) -> None: ...

    async def close(self) -> None: ...


def _log_unexpected_background_exit(task: asyncio.Task[None]) -> None:
    """Surface an UNEXPECTED background-task exit loudly (never silently swallowed).

    A task cancelled during shutdown is expected (no log). A task that finishes on its own while
    the service is still up -- with or without an exception -- means the drainer died under a live
    API (a silent eval outage), so it is logged CRITICAL.
    """
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        _logger.critical("background task exited unexpectedly", exc_info=exc)
    else:
        _logger.critical("background task exited unexpectedly without error")


def create_challenge_app(
    *,
    settings: ChallengeSettings,
    database: ChallengeDatabase,
    public_router: APIRouter,
    get_weights_fn: GetWeightsFn,
    background_tasks: Sequence[BackgroundTaskFactory] = (),
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await database.init()
        tasks: list[asyncio.Task[None]] = [
            asyncio.create_task(factory(app)) for factory in background_tasks
        ]
        for task in tasks:
            task.add_done_callback(_log_unexpected_background_exit)
        try:
            yield
        finally:
            for task in tasks:
                task.cancel()
            for task in tasks:
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    _logger.exception("background task crashed during shutdown")
            await database.close()

    app = FastAPI(title=settings.name, version=settings.version, lifespan=lifespan)

    @app.get("/health", response_model=HealthResponse, include_in_schema=False)
    async def health() -> HealthResponse:
        return HealthResponse(slug=settings.slug, version=settings.version)

    @app.get("/version", response_model=VersionResponse, include_in_schema=False)
    async def version() -> VersionResponse:
        capabilities = ["get_weights", "proxy_routes", "sqlite"]
        backend = getattr(settings, "execution_backend", "")
        if settings.docker_enabled or backend in {
            "base_container",
            "base_gpu",
            "container_gpu",
            "docker_gpu",
        }:
            capabilities.append("docker_executor")
        return VersionResponse(
            api_version=settings.api_version,
            challenge_version=settings.version,
            sdk_version=settings.sdk_version,
            capabilities=capabilities,
        )

    internal_router = APIRouter(
        prefix="/internal/v1",
        dependencies=[Depends(build_internal_auth_dependency(settings))],
    )

    @internal_router.get("/get_weights", response_model=WeightsResponse)
    async def get_weights() -> WeightsResponse:
        weights = await get_weights_fn()
        return WeightsResponse(challenge_slug=settings.slug, epoch=int(time()), weights=weights)

    app.include_router(internal_router)
    app.include_router(public_router)
    return app
