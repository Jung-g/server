from datetime import datetime, timezone
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from core_method import get_db
from models import Token

router = APIRouter()

@router.post("/auto/logout")
def refresh_access_token(refresh_token: str, db: Session = Depends(get_db)):
    token_entry = db.query(Token).filter(Token.Refresh_token == refresh_token).first()

    if not token_entry:
        return {"logged_in": False}

    if token_entry.Expires < datetime.now(timezone.utc):
        db.delete(token_entry)
        db.commit()
        return {"logged_in": False}

    return {"logged_in": True}