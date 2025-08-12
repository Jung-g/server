from fastapi import APIRouter, Depends, HTTPException, Query, Body
from sqlalchemy.orm import Session
from deps_admin import admin_required
from core_method import get_db
from DB_Table import Word

router = APIRouter(prefix="/admin", tags=["admin"])

@router.get("/health/ping")
def ping(_=Depends(admin_required)):
    return {"ok": True}

@router.get("/words")
def list_words(
    q: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
    db: Session = Depends(get_db),
    _=Depends(admin_required),
):
    qs = db.query(Word)
    if q:
        qs = qs.filter(Word.Word.contains(q))
    return qs.order_by(Word.WID.desc()).limit(limit).all()

@router.post("/words", status_code=201)
def create_word(
    word: str = Body(..., embed=True),
    db: Session = Depends(get_db),
    _=Depends(admin_required),
):
    w = Word(Word=word)
    db.add(w); db.commit(); db.refresh(w)
    return w

@router.put("/words/{wid}")
def update_word(
    wid: int,
    word: str = Body(..., embed=True),
    db: Session = Depends(get_db),
    _=Depends(admin_required),
):
    w = db.get(Word, wid)
    if not w:
        raise HTTPException(status_code=404, detail="not found")
    w.Word = word
    db.commit(); db.refresh(w)
    return w

@router.delete("/words/{wid}", status_code=204)
def delete_word(
    wid: int,
    db: Session = Depends(get_db),
    _=Depends(admin_required),
):
    w = db.get(Word, wid)
    if not w:
        raise HTTPException(status_code=404, detail="not found")
    db.delete(w); db.commit()
