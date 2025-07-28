# main.py

# 1. 필요한 모듈과 함수들을 가져옵니다.
# resource_loader를 가장 먼저 import하면, 프로그램 시작 시 모든 리소스가 로딩됩니다.
import re
from .resource_loader import resources
from .calculate_korean2gloss import get_best_translation
from .sbert_kosimcse_module import sbert_kosimcse_search
from .best_similar import find_best_candidate_iterative

def is_jamo_or_numeric_only(text: str) -> bool:
    """
    입력된 텍스트가 한글 자모, 숫자, 공백으로만 구성되어 있는지 확인합니다.
    """
    # ^: 문자열 시작, $: 문자열 끝
    # [ㄱ-ㅎㅏ-ㅣ\d\s]+: 자모(ㄱ-ㅎ, ㅏ-ㅣ), 숫자(\d), 공백(\s)이 하나 이상(+) 있는지 검사
    pattern = r'^[ㄱ-ㅎㅏ-ㅣ\d\s]+$'
    return bool(re.fullmatch(pattern, text))

def process_sentence(text):
    """
    한 문장을 입력받아 표제어 변환의 모든 단계를 처리하고,
    최종 결과를 문자열 리스트로 반환합니다.
    """
    final_sentence_parts = []

    if len(text.split()) == 1:
        
        if text in resources.lemmas_set:
            print(f"'{text}'은 사전에 존재하는 단어입니다.")
            final_sentence_parts.append(text)
            return final_sentence_parts
        else:
            print(f"'{text}'은 사전에 존재하지 않습니다. 유사도 검색 실행")

            search_results = sbert_kosimcse_search([text], topn=3)
            results_for_word = search_results.get(text, {})
            result_details, reason = find_best_candidate_iterative(results_for_word)

            if result_details:
                final_word = result_details['word']
                final_sentence_parts.append(final_word)
                print(f"'{text}' -> '{final_word}' (빈도:{result_details['frequency']}, 점수:{result_details['score']:.4f}) | {reason}")
            else:
                failed_word = f"{text}"
                final_sentence_parts.append(failed_word)
                print(f"'{text}' | {reason}")
                
            return final_sentence_parts

    else:
        # --- [0단계: 1차 변환] ---
        best_result = get_best_translation(text)
        model_name = best_result['model_name']
        translation = best_result['translation']
        avg_similarity = best_result['avg_similarity']
        
        print(f"\n[선택된 모델] {model_name}")
        print(f"[변환 결과] {translation}")
        print(f"[평균 유사도] {avg_similarity:.4f}")
        
        words = translation.split()
        # 중앙 리소스 관리자의 단어 사전을 사용합니다.
        word_status = {word: word in resources.lemmas_set for word in words}
        words_to_search = [word for word, status in word_status.items() if not status]

        # --- [1단계: 사전 단어 확인] ---
        print("\n--- [1단계: 사전 단어 확인 결과] ---")
        print(f"단어별 상태: {word_status}")
        
        # --- [2단계 & 3단계: 유사어 검색 및 최종 선택] ---
        # 사전에 없는 단어가 있을 경우에만 실행합니다.
        if words_to_search:
            print(f"유사어 검색 필요 단어: {words_to_search}")
            search_results = sbert_kosimcse_search(words_to_search, topn=3)

            for word in words:
                if word_status[word]:
                    final_sentence_parts.append(word)
                else:
                    print(f"\n--- '{word}' 유사도 검색 결과 ---")

                    results_for_word = search_results.get(word, {})
                    result_details, reason = find_best_candidate_iterative(results_for_word)

                    if result_details:
                        final_word = result_details['word']
                        final_sentence_parts.append(final_word)
                        print(f"'{word}' -> '{final_word}' (빈도:{result_details['frequency']}, 점수:{result_details['score']:.4f}) | {reason}")
                    else:
                        failed_word = f"{word}"
                        final_sentence_parts.append(failed_word)
                        print(f"'{word}' | {reason}")
            
            return final_sentence_parts
        
        # 사전에 없는 단어가 없을 경우, 1차 변환 결과를 바로 반환합니다.
        else:
            print("\n모든 단어가 사전에 존재하여 추가 검색 없이 변환을 완료합니다.")
            return translation.split()

def main_translate(text: str)-> list:
    final_list = []
    
    if is_jamo_or_numeric_only(text):
        print("입력이 지문자/숫자로만 구성되어 번역을 건너뜁니다.")            
        final_list.append(text)
    else:
        final_list = process_sentence(text)
            
    # 최종 결과를 보기 좋게 출력합니다.
    print("\n" + "="*50)
    print(" 최종 표제어 리스트 ".center(50, "="))
    print(final_list)
    print(f"(타입: {type(final_list)})")
    print("="*50 + "\n")
        
    return final_list