# hong_translate_main.py

from my_jamo import preprocess_input, assemble_jamos, is_jamo_or_numeric_only, contains_jamo
from ollama_translate import ollama_translate_control

def translate_pipeline(raw_text_list: list) -> list:
    """
    하나의 문장을 받아 전체 파이프라인을 실행하고,
    최종적으로 완성된 문장 '하나'를 문자열로 반환합니다.
    """
    if not raw_text_list:
        return []
    
    raw_text = raw_text_list[0]
    
    if is_jamo_or_numeric_only(raw_text):
        print(f"입력된 문장 : {raw_text} -> 자모 또는 숫자만 포함되어 변환을 건너뜁니다.")
        return [raw_text]
    
    else:
    # 1 & 2단계: 자모 조립
        if len(raw_text.split()) == 1:
            print(f"단어가 {raw_text} 하나이므로 원본을 반환합니다.")
            return [raw_text]
        
        if contains_jamo(raw_text):
            print("문장 내에 자모 감지 -> 자모 조립을 수행합니다.\n")
            processed_jamo = preprocess_input(raw_text)
            gloss_sentence = assemble_jamos(processed_jamo)
            print(f" [자모 조립 결과] : {gloss_sentence}\n")
            
            translate_sentence = ollama_translate_control(gloss_sentence)
            print(f"번역 결과 : {translate_sentence}")

            return [translate_sentence]

        else:
            translate_sentence = ollama_translate_control(raw_text)
            print(f"번역 결과 : {translate_sentence}")
            
            return [translate_sentence]