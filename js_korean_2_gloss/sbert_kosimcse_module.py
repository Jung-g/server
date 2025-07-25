# sbert_kosimcse_module.py
from resource_loader import resources # 중앙 리소스 관리자 import
from sentence_transformers import util

def sbert_kosimcse_search(words, topn=3):
    results = {}
    for word in words:
        results[word] = {}
        for model_name, model in resources.SBERT_MODELS.items():
            query_embedding = model.encode(word, convert_to_tensor=True, normalize_embeddings=True)
            corpus_embeddings = resources.SBERT_EMBEDDINGS[model_name]
            
            scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
            top_results = scores.topk(topn)
            
            model_result = []
            for score, idx in zip(top_results.values, top_results.indices):
                model_result.append((resources.lemma_list[int(idx)], round(float(score), 4)))
            
            results[word][model_name] = model_result
            
    return results