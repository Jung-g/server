from fastapi import APIRouter, HTTPException, Body, Depends
from sqlalchemy.orm import Session
from core_method import get_db, create_access_token, create_refresh_token, pwd_context
from DB_Table import User

router = APIRouter(prefix="/auth", tags=["auth"])

@router.post("/login")
def login(
    user_id: str = Body(..., embed=True),
    password: str = Body(..., embed=True),
    db: Session = Depends(get_db),
):
    u = db.get(User, user_id)
    if not u:
        raise HTTPException(status_code=401, detail="invalid credentials")

    ok = False
    # 1) bcrypt 검증 시도
    try:
        ok = pwd_context.verify(password, u.PassWord)
    except Exception:
        ok = False

    # 2) 과거 평문 보관이었다면 게으른 업그레이드
    if not ok and u.PassWord == password:
        u.PassWord = pwd_context.hash(password)
        db.add(u); db.commit(); db.refresh(u)
        ok = True

    if not ok:
        raise HTTPException(status_code=401, detail="invalid credentials")

    access = create_access_token({"sub": u.UserID})
    refresh, _exp = create_refresh_token({"sub": u.UserID})
    return {"access_token": access, "refresh_token": refresh}
