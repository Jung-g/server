# kobart_translate.py
from .resource_loader import resources # 중앙 리소스 관리자 import

def kobart_translate(text: str) -> str:
    """입력 문장을 KoBART로 번역/분석해서 결과 문자열 반환"""
    inputs = resources.kobart_tokenizer(
        text, max_length=25, truncation=True, padding="max_length", return_tensors="pt"
    )
    output_ids = resources.kobart_model.generate(
        input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
        max_length=30, num_beams=5, early_stopping=True
    )
    result = resources.kobart_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return result