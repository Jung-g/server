from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from core_method import get_current_user_id, get_db
from models import StudyRecord

router = APIRouter()
calendar_router = APIRouter()

@calendar_router.get("/study/calendar")
def get_study_records(
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    records = db.query(StudyRecord).filter(StudyRecord.UserID == user_id).all()
    
    result = [
        {
            "date": record.Study_Date.isoformat(), # 이부분 하얀글씨 해결하기
            "sid": record.SID
        }
        for record in records
    ]
    
    return {"records": result}