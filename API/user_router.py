from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from core_method import verify_or_refresh_token, get_db, pwd_context   
from models import BookMark, StudyRecord, Token, User

router = APIRouter()

# 회원가입
class RegisterRequest(BaseModel):
    id: str
    pw: str
    name: str
@router.post("/user/register")
async def register_user(req: RegisterRequest, db: Session = Depends(get_db)):
    id = req.id
    pw = req.pw
    name = req.name
    
    if db.query(User).filter(User.UserID == id).first():
        return {"success": False}

    hashed_pw = pwd_context.hash(pw)
    user = User(UserID=id, PassWord=hashed_pw, UserName=name)
    db.add(user)
    db.commit()
    return {"success": True}

# 회원정보수정(닉네임, 비밀번호 변경)
class UserUpdate(BaseModel):
    name: str | None = None
    new_password: str | None = None
@router.put("/user/update")
async def update_user(update_data: UserUpdate, db: Session = Depends(get_db), user_id: str = Depends(verify_or_refresh_token)):
    user = db.query(User).filter(User.UserID == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")

    updated = False
    if update_data.name:
        user.UserName = update_data.name
        updated = True
    if update_data.new_password:
        user.PassWord = pwd_context.hash(update_data.new_password)
        updated = True

    if updated:
        db.commit()
        return {
            "success": True,
            "nickname": user.UserName
        }
    else:
        return {"success": False}
    
# 비밀번호재설정(로그인 단계에서 비밀번호 기억 안나는 경우)
class PasswordResetRequest(BaseModel):
    user_id: str
    new_password: str
@router.put("/user/reset_password")
async def reset_password(data: PasswordResetRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.UserID == data.user_id).first()
    if not user:
        return {"success": False, "message": "존재하지 않는 ID입니다."}

    user.PassWord = pwd_context.hash(data.new_password)
    db.commit()
    return {"success": True}

# 회원탈퇴
class DeleteUserRequest(BaseModel):
    password: str
@router.post("/delete_user")
def delete_user(data: DeleteUserRequest, db: Session = Depends(get_db), user_id: str = Depends(verify_or_refresh_token)):
    try:
        user = db.query(User).filter(User.UserID == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="유저를 찾을 수 없습니다")

        if not pwd_context.verify(data.password, user.PassWord):
            raise HTTPException(status_code=401, detail="비밀번호가 일치하지 않습니다")

        # DB에서 해당 user 데이터 삭제
        db.query(Token).filter(Token.UserID == user_id).delete()
        db.query(BookMark).filter(BookMark.UserID == user_id).delete()
        db.query(StudyRecord).filter(StudyRecord.UserID == user_id).delete()
        db.delete(user)
        db.commit()

        return {"success": True}
    except Exception as e:
        db.rollback()
        print("오류: ", e)
        return {"success": False}
