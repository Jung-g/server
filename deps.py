from typing import Optional
from fastapi import Depends, HTTPException, Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from core_method import verify_or_refresh_token, get_db
from DB_Table import User

security = HTTPBearer(auto_error=False)

def ensure_bearer(credentials: HTTPAuthorizationCredentials):
    if not credentials or (credentials.scheme or "").lower() != "bearer":
        raise HTTPException(status_code=401, detail="인증 정보가 없습니다.")

def check_user(request: Request, response: Response, db: Session) -> User:
    user_id = verify_or_refresh_token(request, response)
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=401, detail="user not found")
    return user

def user_required(
    request: Request,
    response: Response,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
) -> Optional[User]:
    if request.method == "OPTIONS":
        return None

    ensure_bearer(credentials)
    return check_user(request, response, db)
