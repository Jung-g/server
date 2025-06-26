from fastapi import APIRouter, Body, Depends
from sqlalchemy.orm import Session
from main import get_db
from models import Token

logout_router = APIRouter()

@logout_router.post("/user/logout")
def logout_user(refresh_token: str = Body(...), db: Session = Depends(get_db)):
    token_entry = db.query(Token).filter(Token.Refresh_token == refresh_token).first()

    if not token_entry:
        return {"success": False}

    db.delete(token_entry)
    db.commit()
    return {"success": True}
