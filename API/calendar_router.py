from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from core_method import verify_or_refresh_token, get_db
from models import StudyRecord, Study, StudyStep
from datetime import date, timedelta

router = APIRouter()

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
@router.get("/study/calendar")
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

# 달력 (홈 화면용)
@router.get("/study/stats")
async def get_study_stats(db: Session = Depends(get_db), user_id: str = Depends(verify_or_refresh_token)):
    records = (
        db.query(StudyRecord)
        .filter(
            StudyRecord.UserID == user_id,
            StudyRecord.Complate == True
        )
        .all()
    )

    date_list = [record.Study_Date.date() for record in records]
    learned_dates = [d.isoformat() for d in date_list]
    streak_days = calculate_streak(date_list)

    sid_step_pairs = {(r.SID, r.Step) for r in records}

    word_ids = set()
    for sid, step in sid_step_pairs:
        step_words = (
            db.query(StudyStep.WID)
            .filter(StudyStep.SID == sid, StudyStep.Step == step)
            .all()
        )
        word_ids.update(wid for (wid,) in step_words)

    learned_words_count = len(word_ids)

    completed_steps_by_sid = {}
    for sid, step in sid_step_pairs:
        completed_steps_by_sid.setdefault(sid, []).append(step)

    for sid in completed_steps_by_sid:
        completed_steps_by_sid[sid] = sorted(completed_steps_by_sid[sid])

    return {
        "learned_dates": learned_dates,
        "streak_days": streak_days,
        "learned_words_count": learned_words_count,
        "completed_steps": completed_steps_by_sid
    }
