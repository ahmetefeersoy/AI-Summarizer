from fastapi import APIRouter, Depends, HTTPException
from models import User
from pydantic import BaseModel
from auth import hash_password, verify_password, create_access_token

router = APIRouter(prefix="/users", tags=["Users"])

class SignupRequest(BaseModel):
    email: str
    password: str
    role: str = "AGENT"   

class LoginRequest(BaseModel):
    email: str
    password: str

@router.post("/signup")
async def signup(body: SignupRequest):
    existing = await User.get_or_none(email=body.email)
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    user = await User.create(
        email=body.email,
        password=hash_password(body.password),
        role=body.role
    )
    return {"id": user.id, "email": user.email, "role": user.role}

@router.post("/login")
async def login(body: LoginRequest):
    user = await User.get_or_none(email=body.email)
    if not user or not verify_password(body.password, user.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token({"sub": user.id, "role": user.role})
    return {"access_token": token, "token_type": "bearer"}
