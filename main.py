from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer
from pydantic import BaseModel
from models import User, Note, Role, NoteStatus
from database import init_db, close_db
from auth import hash_password, verify_password, create_access_token, get_current_user, require_admin
from background_jobs import job_manager
from ai_model import ai_model
from datetime import timedelta
from typing import Optional, List
import uuid

app = FastAPI(
    title="AI Summarizer API", 
    description="AI-powered note summarization service.",
    version="1.0.0"
)

security = HTTPBearer()

@app.on_event("startup")
async def startup():
    await init_db()
    await job_manager.start()

@app.on_event("shutdown")
async def shutdown():
    await job_manager.stop()
    await close_db()

class UserCreate(BaseModel):
    email: str
    password: str
    role: Optional[str] = "AGENT" 
class UserLogin(BaseModel):
    email: str
    password: str

class NoteCreate(BaseModel):
    raw_text: str

class NoteResponse(BaseModel):
    id: int
    raw_text: str
    summary: Optional[str]
    status: str
    userId: int
    createdAt: str
    updatedAt: str

class UserResponse(BaseModel):
    id: int
    email: str
    role: str
    createdAt: str

@app.post("/register", response_model=UserResponse)
async def register(user: UserCreate):
    existing = await User.get_or_none(email=user.email)
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    if user.role and user.role not in ["ADMIN", "AGENT"]:
        raise HTTPException(status_code=400, detail="Invalid role. Must be ADMIN or AGENT")

    hashed_pw = hash_password(user.password)
    
    user_role = Role(user.role) if user.role else Role.AGENT
    
    new_user = await User.create(
        email=user.email, 
        password=hashed_pw, 
        role=user_role
    )
    return UserResponse(
        id=new_user.id,
        email=new_user.email,
        role=new_user.role.value,  
        createdAt=new_user.created_at.isoformat()
    )

@app.post("/login", 
          summary="User Login", 
          description="Login with email and password to get access token. Copy the token and use 'Authorize' button in Swagger.")
async def login(user: UserLogin):
    db_user = await User.get_or_none(email=user.email)
    if not db_user or not verify_password(user.password, db_user.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    access_token_expires = timedelta(minutes=60)
    access_token = create_access_token(
        data={"sub": db_user.email, "role": db_user.role.value},  
        expires_delta=access_token_expires
    )
    return {
        "access_token": access_token, 
        "token_type": "bearer",
    }

@app.post("/notes", 
          response_model=NoteResponse,
          summary="Create Note")
async def create_note(note: NoteCreate, current_user = Depends(get_current_user)):
    new_note = await Note.create(
        raw_text=note.raw_text,
        status=NoteStatus.QUEUED,
        user=current_user
    )
    
    job_id = str(uuid.uuid4())
    await job_manager.add_job(
        job_id=job_id,
        job_type="summarize_note",
        data={
            "note_id": new_note.id,
            "raw_text": note.raw_text
        }
    )
    
    return NoteResponse(
        id=new_note.id,
        raw_text=new_note.raw_text,
        summary=new_note.summary,
        status=new_note.status.value,  
        userId=new_note.user_id,
        createdAt=new_note.created_at.isoformat(),
        updatedAt=new_note.updated_at.isoformat()
    )

@app.get("/notes/{note_id}", 
         response_model=NoteResponse,
         summary="Get Note by ID")
async def get_note(note_id: int, current_user = Depends(get_current_user)):
    note = await Note.get_or_none(id=note_id).prefetch_related('user')
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    
    if current_user.role.value == "AGENT" and note.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return NoteResponse(
        id=note.id,
        raw_text=note.raw_text,
        summary=note.summary,
        status=note.status.value,  
        userId=note.user_id,
        createdAt=note.created_at.isoformat(),
        updatedAt=note.updated_at.isoformat()
    )

@app.get("/notes", 
         response_model=List[NoteResponse],
         summary="List User Notes")
async def list_notes(current_user = Depends(get_current_user)):
    
    if current_user.role.value == "AGENT":
        
        notes = await Note.filter(user=current_user).order_by('-created_at')
    else:
        notes = await Note.all().order_by('-created_at')
    
    return [
        NoteResponse(
            id=note.id,
            raw_text=note.raw_text,
            summary=note.summary,
            status=note.status.value,  
            userId=note.user_id,
            createdAt=note.created_at.isoformat(),
            updatedAt=note.updated_at.isoformat()
        )
        for note in notes
    ]

@app.get("/admin/users", response_model=List[UserResponse])
async def list_all_users(current_user = Depends(require_admin)):
    users = await User.all().order_by('-created_at')
    return [
        UserResponse(
            id=user.id,
            email=user.email,
            role=user.role.value,  
            createdAt=user.created_at.isoformat()
        )
        for user in users
    ]

@app.get("/admin/notes", response_model=List[NoteResponse])
async def admin_list_notes(current_user = Depends(require_admin)):
    notes = await Note.all().order_by('-created_at')
    
    return [
        NoteResponse(
            id=note.id,
            raw_text=note.raw_text,
            summary=note.summary,
            status=note.status.value,  
            userId=note.user_id,
            createdAt=note.created_at.isoformat(),
            updatedAt=note.updated_at.isoformat()
        )
        for note in notes
    ]


@app.get("/test-summary")
async def get_test_summary():
    return await ai_model.test_summary()

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/", 
         summary="API Information",
         description="Welcome to AI Summarizer API")
async def root():
    return {
        "message": "AI Summarizer API", 
        "docs": "/docs", 
        "how_to_authenticate": {
            "step_1": "POST /register - Create an account",
            "step_2": "POST /login - Get your access token",
            "step_3": "Click 'Authorize' button in Swagger UI",
            "step_4": "Enter your token (without 'Bearer' prefix)",
            "step_5": "Now you can use protected endpoints"
        },
        "ai_model": ai_model.get_model_info()
    }
