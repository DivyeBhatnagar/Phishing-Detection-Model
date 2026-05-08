"""
PhishGuard AI - Auth Router
=============================
POST /api/v1/auth/register → Create new user
POST /api/v1/auth/login    → Login (returns JWT)
POST /api/v1/auth/refresh  → Refresh access token
GET  /api/v1/auth/me       → Current user info
POST /api/v1/auth/api-key  → Generate API key
"""

from __future__ import annotations

import secrets
from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from backend.core.auth import (
    create_access_token,
    create_refresh_token,
    decode_token,
    get_current_user,
    hash_password,
    verify_password,
)
from backend.db.models import UserDocument
from backend.schemas.schemas import (
    TokenResponse,
    UserLoginRequest,
    UserRegisterRequest,
)
from config.settings import settings
from utils.helpers import utcnow

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post(
    "/register",
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user account",
)
async def register(body: UserRegisterRequest):
    """Create a new user account."""
    # Check uniqueness
    existing = await UserDocument.find_one(
        {"$or": [{"username": body.username}, {"email": body.email}]}
    )
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username or email already registered.",
        )

    user = UserDocument(
        username=body.username,
        email=body.email,
        hashed_password=hash_password(body.password),
    )
    await user.insert()

    return {
        "message": "User registered successfully.",
        "user_id": str(user.id),
        "username": user.username,
    }


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Login and get JWT tokens",
)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Authenticate user and issue JWT tokens."""
    user = await UserDocument.find_one(UserDocument.username == form_data.username)
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials.",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled.",
        )

    # Update last login
    user.last_login = utcnow()
    await user.save()

    token_data = {
        "sub": str(user.id),
        "username": user.username,
        "is_admin": user.is_admin,
    }

    return TokenResponse(
        access_token=create_access_token(token_data),
        refresh_token=create_refresh_token(token_data),
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.post("/refresh", response_model=TokenResponse, summary="Refresh access token")
async def refresh_token(refresh_token_str: str):
    """Use a refresh token to get a new access token."""
    payload = decode_token(refresh_token_str)
    if payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token.",
        )

    token_data = {
        "sub": payload["sub"],
        "username": payload.get("username"),
        "is_admin": payload.get("is_admin", False),
    }

    return TokenResponse(
        access_token=create_access_token(token_data),
        refresh_token=create_refresh_token(token_data),
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.get("/me", summary="Get current user info")
async def get_me(current_user: dict = Depends(get_current_user)):
    """Return current authenticated user info."""
    user = await UserDocument.get(current_user["sub"])
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    return {
        "id": str(user.id),
        "username": user.username,
        "email": user.email,
        "is_admin": user.is_admin,
        "prediction_count": user.prediction_count,
        "created_at": user.created_at,
    }


@router.post("/api-key", summary="Generate or regenerate API key")
async def generate_api_key(current_user: dict = Depends(get_current_user)):
    """Generate a new API key for the authenticated user."""
    user = await UserDocument.get(current_user["sub"])
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    new_key = secrets.token_urlsafe(32)
    user.api_key = new_key
    await user.save()

    return {
        "message": "API key generated. Store this securely — it won't be shown again.",
        "api_key": new_key,
    }
