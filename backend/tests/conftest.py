"""
Shared pytest fixtures.

Environment variables are set before ``app.config`` is imported, so the settings
singleton is built from a known-good test configuration rather than whatever
happens to be in the developer's shell.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

# Must run before any `app.*` import. A module-level assignment in conftest is
# the only reliable hook for that.
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("API_KEY", "test-api-key-000000000000")
os.environ.setdefault("AI_PROVIDER", "rules")
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ.setdefault("LOG_FORMAT", "console")
os.environ.setdefault("LOG_DIR", "")
os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+asyncpg://linkedin:linkedin@localhost:15434/linkedin_agent",
)

from app.config import Settings, get_settings


@pytest.fixture(scope="session")
def settings() -> Settings:
    """Process-wide settings built from the test environment."""
    return get_settings()


@pytest.fixture
def resume_text() -> str:
    """A realistic resume used as ground truth across honesty tests."""
    return """
    Jane Doe — Senior Backend Engineer
    London, UK

    EXPERIENCE
    Acme Corp — Senior Backend Engineer, 2021 - Present
      Built and operated Django services on PostgreSQL serving 2M requests/day.
      Containerised the platform with Docker and deployed on AWS ECS.
      Introduced pytest-based integration testing, cutting production defects 40%.

    Beta Ltd — Backend Engineer, 2019 - 2021
      Python and Flask REST APIs backed by Redis and PostgreSQL.
      Built CI pipelines in GitHub Actions.

    SKILLS
    Python, Django, Flask, PostgreSQL, Redis, Docker, AWS, GitHub Actions, pytest, Git

    EDUCATION
    BSc Computer Science, University of Manchester, 2019
    """.strip()


@pytest.fixture
def job_description() -> str:
    """A job posting deliberately containing technologies absent from the resume."""
    return """
    We are hiring a Senior Backend Engineer.

    Requirements:
      - Strong Python, ideally with Django or FastAPI
      - PostgreSQL and Redis in production
      - Docker and Kubernetes for orchestration
      - Experience with Kafka event streaming
      - AWS or GCP
      - Terraform for infrastructure as code

    Nice to have: Go, gRPC, an AWS certification.
    """.strip()


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _service_available(url: str) -> bool:
    """True if a TCP connection to the host/port in ``url`` succeeds."""
    import socket
    from urllib.parse import urlparse

    parsed = urlparse(url.replace("postgresql+asyncpg", "postgresql"))
    if not parsed.hostname or not parsed.port:
        return False
    try:
        with socket.create_connection((parsed.hostname, parsed.port), timeout=2):
            return True
    except OSError:
        return False


@pytest.fixture(scope="session")
def postgres_available(settings: Settings) -> bool:
    return _service_available(settings.database_url)


@pytest.fixture(autouse=True)
def _skip_integration_without_services(
    request: pytest.FixtureRequest, settings: Settings
) -> Iterator[None]:
    """Skip ``@pytest.mark.integration`` tests when Postgres is not reachable.

    Keeps `pytest` green on a laptop with nothing running, while the same suite
    runs fully in CI where the service container is up.
    """
    if request.node.get_closest_marker("integration") and not _service_available(
        settings.database_url
    ):
        pytest.skip("Postgres is not reachable; skipping integration test")
    yield
