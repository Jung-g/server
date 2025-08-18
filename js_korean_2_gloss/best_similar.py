from collections import defaultdict

def find_best_candidate_iterative(search_results_for_word):
    """
    빈도수 5 -> 1 순서로 순회하며 계층적 규칙에 따라 최적의 대체 단어를 선택합니다.
    사용자가 임계값 조정을 위해 각 단계의 평균 점수를 확인할 수 있도록 상세 정보를 출력합니다.
    """
    # 1. [전처리] 단어별 빈도, 평균 점수, 빈도수 그룹을 미리 계산합니다.
    scores_by_word = defaultdict(list)
    for result_list in search_results_for_word.values():
        for word, score in result_list:
            scores_by_word[word].append(score)

    if not scores_by_word:
        return None, "최종 실패 (검색 결과 없음)"

    aggregated_stats = {
        word: {"frequency": len(scores), "avg_score": sum(scores) / len(scores)}
        for word, scores in scores_by_word.items()
    }

    words_by_freq = defaultdict(list)
    for word, stats in aggregated_stats.items():
        words_by_freq[stats['frequency']].append((word, stats['avg_score']))

    # 2. [순회] 빈도수 5부터 2까지 검사 (기준: 평균 점수 >= 0.75)
    for freq in range(5, 1, -1):
        # 해당 빈도수의 후보가 없으면 다음 단계로 넘어갑니다.
        if not words_by_freq[freq]:
            continue
        
        sorted_candidates = sorted(words_by_freq[freq], key=lambda x: x[1], reverse=True)

        print(f"\n[ 빈도수 {freq} 검사 ]")
        freq_candidates_str = ", ".join([f"'{w}'(평균:{s:.4f})" for w, s in sorted_candidates])
        print(f"  - 후보: {freq_candidates_str}")

        # 기준을 통과하는 후보들을 찾습니다.
        passed_candidates = [(w, s) for w, s in sorted_candidates if s >= 0.7]

        if passed_candidates:
            best_word, best_score = passed_candidates[0]
            result_details = {'word': best_word, 'score': best_score, 'frequency': freq}
            return result_details, f"성공 (빈도: {freq}, 평균점수≥0.75)"

    # 3. [빈도수 1 검사] (기준: 절대 점수 >= 0.8)
    if words_by_freq[1]:
        print(f"\n[ 빈도수 1 검사 ]")
        
        sorted_candidates = sorted(words_by_freq[1], key=lambda x: x[1], reverse=True)

        freq_candidates_str = ", ".join([f"'{w}'(점수:{s:.4f})" for w, s in sorted_candidates])
        print(f"  - 후보: {freq_candidates_str}")

        passed_candidates = [(w, s) for w, s in sorted_candidates if s >= 0.7]

        if passed_candidates:
            best_word, best_score = passed_candidates[0]
            result_details = {'word': best_word, 'score': best_score, 'frequency': 1}
            return result_details, "성공 (빈도: 1, 절대점수≥0.8)"

    # 4. [최종 선택] 모든 기준 실패 시, 빈도수 5였던 단어 중 최고점자 선택
    print("\n[ 모든 기준 미충족. 최종 선택(Fallback) 시도 ]")
    if words_by_freq[5]:
        print("  - 빈도수 5였던 단어 중 평균 점수가 가장 높은 단어를 채택합니다.")
        words_by_freq[5].sort(key=lambda x: x[1], reverse=True)
        best_word, best_score = words_by_freq[5][0]
        result_details = {'word': best_word, 'score': best_score, 'frequency': 5}
        return result_details, "최종 선택 (빈도수 5 중 최고점)"

    return None, "최종 실패 (모든 기준 및 최종 선택 실패)"