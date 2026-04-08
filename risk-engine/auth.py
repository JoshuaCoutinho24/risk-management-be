"""
JWT-based authentication for RiskEngine Pro.

Single-user setup:
    username : admin   (override via ADMIN_USERNAME env var)
    password : Js@8C2024  (override via ADMIN_PASSWORD env var)

All tokens are HS256 JWTs with a 24-hour expiry.
Uses bcrypt directly (no passlib) to avoid the bcrypt 5.x incompatibility.
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from pydantic import BaseModel

# ── Config ────────────────────────────────────────────────────────────────────

_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "riskengine-super-secret-change-in-prod-2024")
_ALGORITHM  = "HS256"
_EXPIRE_HOURS = 24

_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
_PASSWORD = os.environ.get("ADMIN_PASSWORD", "Js@8C2024")

# Hash the password once at module load time
_HASHED: bytes = bcrypt.hashpw(_PASSWORD.encode(), bcrypt.gensalt())

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


# ── Pydantic models ───────────────────────────────────────────────────────────

class Token(BaseModel):
    access_token: str
    token_type:   str = "bearer"


class TokenData(BaseModel):
    username: Optional[str] = None


# ── Core helpers ──────────────────────────────────────────────────────────────

def _verify_password(plain: str, hashed: bytes) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed)


def authenticate_user(username: str, password: str) -> Optional[str]:
    """Return the username on success, None on failure.

    Always runs a bcrypt check (even for wrong usernames) to prevent
    timing-based user enumeration.
    """
    # Always run bcrypt so response time is constant
    if username == _USERNAME:
        ok = _verify_password(password, _HASHED)
    else:
        # Dummy check — prevent enumeration via timing
        bcrypt.checkpw(b"dummy", _HASHED)
        ok = False

    return username if ok else None


def create_access_token(subject: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(hours=_EXPIRE_HOURS)
    payload = {"sub": subject, "exp": expire}
    return jwt.encode(payload, _SECRET_KEY, algorithm=_ALGORITHM)


# ── FastAPI dependency ────────────────────────────────────────────────────────

async def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, _SECRET_KEY, algorithms=[_ALGORITHM])
        username: Optional[str] = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    if username != _USERNAME:
        raise credentials_exception

    return username
