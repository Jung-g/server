# calculate_korean2gloss.py

from resource_loader import resources # 중앙 리소스 관리자는 이미 SBERT 모델들을 로드하고 있습니다.
from sentence_transformers import util

def calculate_similarity(sentence1: str, sentence2: str) -> float:
    """
    주어진 두 문장(sentence1, sentence2) 사이의 평균 코사인 유사도를 계산합니다.
    resource_loader에 로드된 모든 SBERT 모델의 유사도 점수를 평균내어 반환합니다.
    """
    total_similarity = 0.0
    num_models = len(resources.SBERT_MODELS)

    # 모델이 로드되지 않은 경우 0을 반환
    if num_models == 0:
        return 0.0

    # 모든 SBERT 모델을 순회하며 유사도 계산
    for model in resources.SBERT_MODELS.values():
        # 두 문장을 각각 임베딩(벡터)으로 변환
        emb1 = model.encode(sentence1, convert_to_tensor=True, normalize_embeddings=True)
        emb2 = model.encode(sentence2, convert_to_tensor=True, normalize_embeddings=True)

        # 코사인 유사도 계산 후 총합에 더하기
        similarity_score = util.cos_sim(emb1, emb2).item()
        total_similarity += similarity_score

    # 전체 유사도 합계를 모델 수로 나누어 평균 계산
    average_similarity = total_similarity / num_models
    return average_similarity