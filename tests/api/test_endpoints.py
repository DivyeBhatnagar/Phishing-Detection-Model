"""
PhishGuard AI - API Integration Tests
=======================================
Tests for FastAPI endpoints using httpx async test client.
"""

import pytest
from httpx import AsyncClient, ASGITransport

# Mark all tests as async
pytestmark = pytest.mark.asyncio


@pytest.fixture
async def client():
    """Create async test client for FastAPI app."""
    from backend.main import app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestHealthEndpoint:
    """Tests for GET /api/v1/health."""

    async def test_health_returns_200(self, client: AsyncClient):
        response = await client.get("/api/v1/health")
        assert response.status_code == 200

    async def test_health_response_schema(self, client: AsyncClient):
        response = await client.get("/api/v1/health")
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "model_loaded" in data
        assert "database_connected" in data
        assert "uptime_seconds" in data

    async def test_health_status_values(self, client: AsyncClient):
        response = await client.get("/api/v1/health")
        data = response.json()
        assert data["status"] in ["healthy", "degraded", "unhealthy"]


class TestPredictEndpoint:
    """Tests for POST /api/v1/predict."""

    async def test_predict_rejects_empty_email(self, client: AsyncClient):
        response = await client.post(
            "/api/v1/predict",
            json={"email_text": ""}
        )
        assert response.status_code == 422  # Pydantic validation error

    async def test_predict_rejects_too_short_email(self, client: AsyncClient):
        response = await client.post(
            "/api/v1/predict",
            json={"email_text": "Hi"}
        )
        assert response.status_code == 422

    async def test_predict_schema_validation(self, client: AsyncClient):
        """If model is loaded, response should match schema."""
        response = await client.post(
            "/api/v1/predict",
            json={
                "email_text": "Congratulations! You have won a prize. Click here to claim."
            }
        )
        # Either 200 (model loaded) or 503 (model not yet trained)
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "prediction" in data
            assert data["prediction"] in ["spam", "legitimate"]
            assert "confidence" in data
            assert 0 <= data["confidence"] <= 100
            assert "risk_level" in data
            assert data["risk_level"] in ["low", "medium", "high"]

    async def test_predict_suspicious_input_rejected(self, client: AsyncClient):
        """Suspicious/injection input should be rejected."""
        response = await client.post(
            "/api/v1/predict",
            json={"email_text": "<script>alert('xss')</script> This is an email"}
        )
        assert response.status_code == 400


class TestRootEndpoint:
    """Tests for GET /."""

    async def test_root_returns_info(self, client: AsyncClient):
        response = await client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "docs" in data


class TestDocsEndpoints:
    """Swagger / ReDoc documentation endpoints."""

    async def test_swagger_docs_available(self, client: AsyncClient):
        response = await client.get("/api/docs")
        assert response.status_code == 200

    async def test_openapi_schema_available(self, client: AsyncClient):
        response = await client.get("/api/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "paths" in data
