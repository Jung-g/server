# calculate_korean2gloss.py
from resource_loader import resources # 중앙 리소스 관리자 import
from kobart_translate import kobart_translate
from KoT5_translate import kot5_translate
from sentence_transformers import util

def get_best_translation(sentence: str):
    kobart_out = kobart_translate(sentence)
    kot5_out  = kot5_translate(sentence)

    sim_sums = {"kobart": 0.0, "kot5": 0.0}
    # 리소스 관리자에 로드된 SBERT 모델들을 사용합니다.
    for model in resources.SBERT_MODELS.values():
        emb_o = model.encode(sentence, convert_to_tensor=True, normalize_embeddings=True)
        emb_b = model.encode(kobart_out, convert_to_tensor=True, normalize_embeddings=True)
        emb_t = model.encode(kot5_out, convert_to_tensor=True, normalize_embeddings=True)
        sim_sums["kobart"] += util.cos_sim(emb_o, emb_b).item()
        sim_sums["kot5"]   += util.cos_sim(emb_o, emb_t).item()

    n = len(resources.SBERT_MODELS)
    avg_kobart = sim_sums["kobart"] / n
    avg_kot5   = sim_sums["kot5"]   / n

    if avg_kobart >= avg_kot5:
        return {"model_name": "KoBART", "translation": kobart_out, "avg_similarity": avg_kobart}
    else:
        return {"model_name": "KoT5", "translation": kot5_out, "avg_similarity": avg_kot5}