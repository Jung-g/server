from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from core_method import verify_or_refresh_token, get_db
from models import StudyRecord

router = APIRouter()

@router.get("/study/calendar")
async def get_study_records(db: Session = Depends(get_db), user_id: str = Depends(verify_or_refresh_token)):
    records = (
        db.query(StudyRecord)
        .filter(StudyRecord.UserID == user_id, StudyRecord.Complate == True)
        .all()
    )

    result = [record.Study_Date.date().isoformat() for record in records]

    return {"records": result}