from datetime import datetime
import os
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
import httpx
from pydantic import BaseModel
from sqlalchemy.orm import Session
from core_method import get_db, verify_or_refresh_token
from models import Word, WordDetail

router = APIRouter()

load_dotenv(dotenv_path="dict_api_key.env")
OPEN_DICT_KEY = os.getenv("OPEN_DICT_KEY")
CERTKEY_NO = os.getenv("CERTKEY_NO")

# db에 저장된 단어
@router.get("/dictionary/words")
async def get_words(request: Request, response: Response, query: str = Query(None), db: Session = Depends(get_db),):
    user_id = verify_or_refresh_token(request, response)

    if query:
        words = db.query(Word).filter(Word.Word.contains(query)).all()
    else:
        words = db.query(Word).all()

    result = []
    for w in words:
        # animation_path = w.animations[0].AnimePath if w.animations else ""
        result.append({
            "wid": w.WID,
            "word": w.Word,
        })

    return result

# 단어 상세정보
@router.get("/dictionary/words/detail")
async def get_words_detail(wid: int = Query(..., description="Word의 WID"), db: Session = Depends(get_db), user_id: str = Depends(verify_or_refresh_token)):
    try:
        word_obj = db.query(Word).get(wid)
        if not word_obj or not word_obj.Word:
            raise HTTPException(status_code=404, detail="해당 단어는 DB에 존재하지 않습니다.")
        word_text = word_obj.Word.strip()

        detail = db.query(WordDetail).get(wid)
        if detail:
            return {
                "word": word_text,
                "pos": detail.Pos,
                "definition": detail.Definition,
            }

        if not OPEN_DICT_KEY:
            raise HTTPException(status_code=500, detail="API 키 오류")

        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(
                "https://opendict.korean.go.kr/api/search",
                params={
                    "certkey_no": CERTKEY_NO,
                    "key": OPEN_DICT_KEY,
                    "target_type": "search",
                    "req_type": "json",
                    "part": "word",
                    "q": word_text,
                    "sort": "dict",
                    "start": 1,
                    "num": 10,
                }
            )

        if r.status_code != 200:
            raise HTTPException(status_code=500, detail=f"API 호출 실패: {r.status_code}")

        try:
            data = r.json()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"JSON 파싱 실패: {str(e)}")

        items = data.get("channel", {}).get("item")
        if isinstance(items, dict):
            items = [items]
        if not isinstance(items, list) or not items:
            raise HTTPException(status_code=404, detail="검색 결과 없음")

        first_item = items[0]
        sense_list = first_item.get("sense", [])
        if isinstance(sense_list, dict):
            sense_list = [sense_list]
        if not sense_list:
            raise HTTPException(status_code=404, detail="뜻 정보 없음")

        first_sense = sense_list[0]
        pos = first_sense.get("pos", "")
        definition = first_sense.get("definition", "")

        new_detail = WordDetail(
            WID=word_obj.WID,
            Pos=pos,
            Definition=definition,
            UpdatedTime=datetime.now()
        )
        db.add(new_detail)
        db.commit()

        return {
            "word": word_text,
            "pos": pos,
            "definition": definition,
        }

    except Exception:
        import traceback
        print("예외 발생:", traceback.format_exc())
        raise HTTPException(status_code=500, detail="서버 내부 오류 발생")