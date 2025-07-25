# gemini_pro_prompt.py
import google.generativeai as genai

# 이 모듈에서 사용할 모델 이름을 상수로 정의합니다.
_MODEL_NAME = 'gemini-2.5-pro'

def get_pro_translation(original_sentence: str, first_gloss: str) -> str:
    """
    원본 문장과 1차 표제어를 받아 Gemini Pro API로 재변환을 요청하고,
    그 결과를 문자열로 반환합니다. (한도 계산 로직 제거됨)
    """
    # 1. 입력받은 문장으로 프롬프트를 동적으로 생성합니다.
    prompt = (
        f"원문 : {original_sentence}\n"
        f"1차 한국 문장 번역 : {first_gloss}\n"
        "위 두 문장의 의미 연관성이 낮다고 판단되어, 원문을 자연스러운 한국어 문맥으로 재변환한 문장 하나만 반환해주세요."
    )
    
    try:
        # 2. API를 호출합니다.
        model = genai.GenerativeModel(_MODEL_NAME)
        generation_config = genai.GenerationConfig(temperature=0.0)
        response = model.generate_content(contents=prompt, generation_config=generation_config)
        
        # 3. API 응답을 분석하여 결과 또는 특정 오류 메시지를 반환합니다.
        if response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'finish_reason'):
                if candidate.finish_reason == 2:  # MAX_TOKENS
                    return "[Error] 응답이 너무 길어 잘렸습니다. max_output_tokens를 늘려주세요."
                elif candidate.finish_reason == 3:  # SAFETY
                    return "[Error] 안전 필터에 의해 차단되었습니다."
                elif candidate.finish_reason == 4:  # RECITATION
                    return "[Error] 저작권 콘텐츠 인용으로 차단되었습니다."
        
        # 정상적인 텍스트 추출
        if hasattr(response, 'text') and response.text:
            return response.text
        else:
            return "[Error] 빈 응답이 반환되었습니다."
        
    # ❹ **(개선)** API 호출 시 발생할 수 있는 예외를 처리합니다.
    except Exception as e:
        print(f"[API Error] API 호출 중 오류가 발생했습니다: {e}")
        return f"[Error] API 호출 실패: {e}"