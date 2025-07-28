# controller.py

# [수정] os, genai 관련 import 및 설정 코드 모두 제거
from . import gemini_pro_prompt
from . import gemini_flash_prompt
    
def gemini_translate_control(original_sentence: str, first_gloss: str):
    """
    Pro 모델을 먼저 시도하고, 에러 발생 시 Flash 모델을 시도하는 전체 흐름을 제어합니다.
    """
    print("--- 1. Gemini Pro 모델 호출 시도 ---")
    pro_result = gemini_pro_prompt.get_pro_translation(original_sentence, first_gloss)
    
    if not pro_result.startswith("[Error"):
        print("Pro 모델 호출 성공!")
        return pro_result
    
    print(f"⚠️ Pro 모델 호출 실패: {pro_result}")
    print("\n--- 2. Gemini Flash 모델 호출 시도 ---")
    
    flash_result = gemini_flash_prompt.get_flash_translation(original_sentence, first_gloss)
    
    if not flash_result.startswith("[Error"):
        print("Flash 모델 호출 성공!")
        return flash_result
    
    print(f"Flash 모델도 호출 실패: {flash_result}")
    return flash_result