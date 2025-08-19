from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from core_method import verify_or_refresh_token, get_db
from DB_Table import StudyRecord, Study, StudyStepMeta
from datetime import date, datetime, time, timedelta

router = APIRouter(tags=["Calendar"])

def calculate_streak(dates: list[date]) -> int:
    today = date.today()
    streak = 0
    check = today

    date_set = set(dates)
    while check in date_set:
        streak += 1
        check -= timedelta(days=1)

    return streak

def calculate_best_streak(dates: list[date]) -> int:
    if not dates:
        return 0
    if len(dates) == 1:
        return 1

    sorted_dates = sorted(set(dates))
    max_streak = 1
    current_streak = 1

    for i in range(1, len(sorted_dates)):
        if (sorted_dates[i] - sorted_dates[i - 1]).days == 1:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 1

    return max_streak

# 달력
@router.get("/study/calendar", summary="학습 기록에 대한 정보를 가져옵니다.", description="학습 기록에 대한 정보를 가져옵니다.")
async def get_study_records(db: Session = Depends(get_db), user_id: str = Depends(verify_or_refresh_token)):
    records = (
        db.query(StudyRecord)
        .filter(StudyRecord.UserID == user_id, StudyRecord.Complate == True)
        .all()
    )

    date_list = [record.Study_Date.date() for record in records]
    learned_dates = [d.isoformat() for d in date_list]
    streak_days = calculate_streak(date_list)
    best_streak = calculate_best_streak(date_list)
    print(best_streak)
    return {
        "records": learned_dates,
        "streak_days": streak_days,
        "best_streak": best_streak
    }

# 특정 날짜 학습 기록 조회
@router.get("/study/records/day", summary="선택한 날짜에 대한 학습 기록의 상세정보를 가져옵니다.", description="선택한 날짜에 대한 학습 기록의 상세정보를 가져옵니다.")
async def get_day_records(date_str: str = Query(..., description="YYYY-MM-DD 형식"), db: Session = Depends(get_db), user_id: str = Depends(verify_or_refresh_token),):
    try:
        selected_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return {"date": date_str, "items": []}

    start_dt = datetime.combine(selected_date, time.min)
    end_dt = start_dt + timedelta(days=1)

    rows = (
        db.query(
            StudyRecord.SID,
            Study.Study_Course,
            StudyRecord.Step,
            StudyStepMeta.StepName,
            StudyRecord.Study_Date,
            StudyRecord.Complate,
        )
        .join(Study, Study.SID == StudyRecord.SID)
        .outerjoin(
            StudyStepMeta,
            (StudyStepMeta.SID == StudyRecord.SID)
            & (StudyStepMeta.Step == StudyRecord.Step),
        )
        .filter(
            StudyRecord.UserID == user_id,
            StudyRecord.Study_Date >= start_dt,
            StudyRecord.Study_Date < end_dt,
        )
        .order_by(StudyRecord.Study_Date.asc())
        .all()
    )

    items = [
        {
            "sid": r.SID,
            "study_course": r.Study_Course,
            "step": r.Step,
            "step_name": r.StepName,
            "study_time": r.Study_Date.isoformat(),
            "complete": bool(r.Complate),
        }
        for r in rows
    ]

    return {"date": date_str, "items": items}