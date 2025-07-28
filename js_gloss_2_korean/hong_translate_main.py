# hong_translate_main.py

from js_korean_2_gloss.resource_loader import resources
from .my_jamo import preprocess_input, assemble_jamos
from .calculate_korean2gloss import calculate_similarity
from .gemini_control import gemini_translate_control

def translate_pipeline(raw_text: str) -> list:
    """
    하나의 문장을 받아 전체 파이프라인을 실행하고,
    최종적으로 완성된 문장 '하나'를 문자열로 반환합니다.
    """
    
    final_sentence_list = []
    
    # 1 & 2단계: 자모 조립
    processed_jamo = preprocess_input(raw_text)
    gloss_sentence = assemble_jamos(processed_jamo)
    print(f" [자모 조립 결과] {gloss_sentence}")

    # 3단계: 표제어 -> KoBART 변환
    kobart_sentence = gloss_to_natural_sentence(gloss_sentence)
    print(f" [KoBART 변환] {kobart_sentence}")
    
    # 4단계: 유사도 검사
    similarity = calculate_similarity(gloss_sentence, kobart_sentence)
    print(f" [유사도 검사] {similarity:.4f}")

    # 기본값은 KoBART 결과로 설정
    final_sentence = kobart_sentence

    # 5단계: 유사도 점수에 따른 Gemini 호출
    if similarity <= 0.79:
        print("-" * 60)
        print(f"유사도({similarity:.4f})가 0.8 이하이므로 Gemini API를 호출하여 재번역합니다.")
        gemini_result = gemini_translate_control(raw_text, gloss_sentence)
        
        if not gemini_result.startswith("[Error"):
            final_sentence = gemini_result
        else:
            print(f"Gemini 호출 실패! KoBART 결과를 최종 결과로 사용합니다.")
        
    final_sentence_list.append(final_sentence)

    print(f"최종 반환 데이터 : {final_sentence_list}")
    print(f"(타입: {type(final_sentence_list)})")
    
    return final_sentence_list

def gloss_to_natural_sentence(gloss_text: str) -> str:
    """KoBART 모델로 표제어 문장을 자연어 문장으로 변환합니다."""
    inputs = resources.kobart_g2s_tokenizer(gloss_text, return_tensors="pt", max_length=128, truncation=True)
    output_tokens = resources.kobart_g2s_model.generate(
        inputs.input_ids, max_length=128, num_beams=5, early_stopping=True
    )
    return resources.kobart_g2s_tokenizer.decode(output_tokens[0], skip_special_tokens=True)