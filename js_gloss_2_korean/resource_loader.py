# resource_loader.py

import os
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast

class ResourceManager:
    def __init__(self):
        print("="*50)
        print("중앙 리소스 관리자 초기화를 시작합니다...")

        # --- 1. 장치 및 경로 설정 ---
        # 경로 설정 (변경 없음)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(current_dir, 'dataset')
        
        self.PATHS = {
            "kobart_g2s_model": os.path.join(models_dir, "kobart_lemma_to_sentence_model"),
        }

        # --- 2. 모델 로딩 ---
        print("\n[1/2] [표제어->문장] KoBART 모델 로딩...")
        self.kobart_g2s_tokenizer = PreTrainedTokenizerFast.from_pretrained(self.PATHS["kobart_g2s_model"])
        # [수정] .to(self.DEVICE) 부분을 삭제합니다.
        self.kobart_g2s_model = BartForConditionalGeneration.from_pretrained(self.PATHS["kobart_g2s_model"])

        print("\n[2/3] SBERT 계열 모델 5종 로딩...")
        self.SBERT_MODELS = { "sbert": SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS"), "kosim_roberta": SentenceTransformer("BM-K/KoSimCSE-roberta"), "kosim_multitask": SentenceTransformer("BM-K/KoSimCSE-RoBERTa-multitask"), "bge_m3_korean": SentenceTransformer("upskyy/bge-m3-korean"), "ko_sroberta_multitask": SentenceTransformer("jhgan/ko-sroberta-multitask") }
        
        # --- [추가] 3. Gemini API 설정 ---
        print("\n[3/3] Gemini API 설정...")
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            print("  ✅ Gemini API 키가 성공적으로 설정되었습니다.")
        else:
            print("  ⚠️ 경고: GEMINI_API_KEY 환경변수가 설정되지 않았습니다. Gemini 호출이 실패할 수 있습니다.")

        print("\n모든 리소스 로딩 완료!")
        print("="*50)

# 프로그램 전체에서 사용될 단 하나의 리소스 관리자 인스턴스를 생성합니다.
resources = ResourceManager()