# api/auth.py
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
import os, jwt
from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel, EmailStr
from passlib.hash import bcrypt
from services.db import get_conn

JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-me")
JWT_EXPIRES_MIN = int(os.getenv("JWT_EXPIRES_MIN", "120"))

router = APIRouter(prefix="/auth", tags=["auth"])

class RegisterIn(BaseModel):
    email: EmailStr
    password: str
    name: Optional[str] = None
    org: Optional[str] = None

class LoginIn(BaseModel):
    email: EmailStr
    password: str

def _make_token(payload: Dict[str, Any]) -> str:
    exp = datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRES_MIN)
    payload = {**payload, "exp": exp}
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")

def current_user(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing token")
    token = authorization.split(" ", 1)[1].strip()
    try:
        data = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    # fetch user
    with get_conn() as con:
        u = con.execute("SELECT id, email, name, org, api_key FROM users WHERE id = ?", (data["uid"],)).fetchone()
        if not u:
            raise HTTPException(status_code=401, detail="User not found")
        return dict(u)

@router.post("/register")
def register(body: RegisterIn):
    pw_hash = bcrypt.hash(body.password)
    with get_conn() as con:
        try:
            con.execute(
                "INSERT INTO users(email, password_hash, name, org) VALUES(?,?,?,?)",
                (body.email, pw_hash, body.name, body.org)
            )
            uid = con.execute("SELECT last_insert_rowid() id").fetchone()["id"]
        except Exception:
            raise HTTPException(status_code=400, detail="Email already registered")
    token = _make_token({"uid": uid})
    return {"ok": True, "token": token}

@router.post("/login")
def login(body: LoginIn):
    with get_conn() as con:
        u = con.execute("SELECT id, password_hash FROM users WHERE email = ?", (body.email,)).fetchone()
        if not u or not bcrypt.verify(body.password, u["password_hash"]):
            raise HTTPException(status_code=401, detail="Invalid credentials")
    token = _make_token({"uid": u["id"]})
    return {"ok": True, "token": token}

@router.get("/me")
def me(user = Depends(current_user)):
    return {"ok": True, "user": user}