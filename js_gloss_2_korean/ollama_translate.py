import ollama
from js_korean_2_gloss.resource_loader import resources  

def ollama_translate_control(original_sentence: str) -> str:
    """
    Ollama로 실행된 로컬 LLM을 호출하여 문장을 변환합니다.
    """
    prompt = (
        "너는 수어 표제어로 이루어진 문장을 한국어 문장으로 변환하는 전문가야.\n"
        f"원문: {original_sentence}\n"
        "수어 표제어라는 특수한 원문의 문맥을 고려하여 자연스러운 문장으로 변환해줘.\n"
        "원문을 변환할 때 표제어를 생략하지 않고 주어와 목적어가 명확하게 드러나게 해줘.\n"
        "한국어 문맥과 문법적으로 올바른 한국어 문장을 만들어줘.\n"
        "수어 표제어 특성상 시점이 불분명할 수 있으니, 문맥에 맞는 시제를 사용해줘.\n"
        "수어 표제어로 이루어진 원문을 자연스러운 한국어 문장으로 재변환한 문장만 반환해줘.\n"
        "변환된 문장을 반환할 때 따옴표는 제거해줘.\n"
        "어떤 경우에도 원문을 반환하지 말고, 반드시 변환이 완료된 문장만 반환해줘"
    )
    
    try:
        response = ollama.chat(
            model=resources.OLLAMA_MODEL_NAME,
            messages=[{'role': 'user', 'content': prompt}]
        )
        result = response['message']['content'].strip(' "\'')
        return result
    except Exception as e:
        error_message = f"[Error] 로컬 모델 호출 실패: {e}"
        print(f"{error_message}")
        return error_message