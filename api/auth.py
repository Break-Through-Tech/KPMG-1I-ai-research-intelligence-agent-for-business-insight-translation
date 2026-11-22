"""Firebase sign-in check for protected routes.

Verifying a token needs only the Firebase project ID: no service-account key, no billing.
"""

import os
import threading
from typing import Any

import firebase_admin
from dotenv import load_dotenv
from fastapi import Header, HTTPException
from firebase_admin import auth

load_dotenv()  # finds the repo-root .env, so FIREBASE_PROJECT_ID works under `make dev`

_init_lock = threading.Lock()


def _ensure_app() -> None:
    # Initialised on the first request, so importing api.main (health check, tests) never needs the env var.
    with _init_lock:
        try:
            firebase_admin.get_app()
        except ValueError:
            project_id = os.getenv("FIREBASE_PROJECT_ID")
            if not project_id:
                raise HTTPException(status_code=500, detail="Server auth is not configured (FIREBASE_PROJECT_ID).")
            firebase_admin.initialize_app(options={"projectId": project_id})




def require_kpmg_user(authorization: str = Header(default="")) -> dict[str, Any]:
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=401, detail="Please sign in.")
    _ensure_app()
    try:
        claims = auth.verify_id_token(token, clock_skew_seconds=10)
    except Exception:
        raise HTTPException(status_code=401, detail="Your session is invalid or expired. Please sign in again.")
    if not claims.get("email_verified"):
        raise HTTPException(status_code=403, detail="Please verify your email first.")
    return claims