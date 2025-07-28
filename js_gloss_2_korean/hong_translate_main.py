# hong_translate_main.py

import time
from my_jamo import preprocess_input, assemble_jamos, is_jamo_or_numeric_only
from ollama_translate import ollama_translate_control

def translate_pipeline(raw_text: str) -> list:
    """
    하나의 문장을 받아 전체 파이프라인을 실행하고,
    최종적으로 완성된 문장 '하나'를 문자열로 반환합니다.
    """
    final_sentence_list = []
    
    if is_jamo_or_numeric_only(raw_text):
        print("입력된 문장이 자모 또는 숫자만 포함되어 있습니다. 변환을 건너뜁니다.")
        final_sentence_list.append(raw_text)        
        return final_sentence_list
    
    else:
    # 1 & 2단계: 자모 조립
        processed_jamo = preprocess_input(raw_text)
        gloss_sentence = assemble_jamos(processed_jamo)
        print(f" [자모 조립 결과] {gloss_sentence}")

        if len(gloss_sentence.split()) == 1:
            print("단어가 하나이므로 추가 변환을 건너뛰고 원본을 반환합니다.")
            final_sentence_list.append(gloss_sentence)
            return final_sentence_list
            
        final_sentence_list.append(ollama_translate_control(gloss_sentence))
    
    return final_sentence_list