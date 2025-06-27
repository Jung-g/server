from datetime import datetime, timezone
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from core_method import get_db
from models import Token

router = APIRouter()

class RefreshRequest(BaseModel):
    refresh_token: str

@router.post("/auto/logout")
async def refresh_access_token(req: RefreshRequest, db: Session = Depends(get_db)):
    refresh_token = req.refresh_token
    token_entry = db.query(Token).filter(Token.Refresh_token == refresh_token).first()

    if not token_entry:
        return {"logged_in": False}

    expires = token_entry.Expires.replace(tzinfo=timezone.utc)

    if expires < datetime.now(timezone.utc):
        db.delete(token_entry)
        db.commit()
        return {"logged_in": False}

    return {"logged_in": True}