import os
import csv
from sentence_transformers import SentenceTransformer
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast, T5ForConditionalGeneration, T5TokenizerFast
import torch
import ollama  
import sys     

class ResourceManager:
    def __init__(self):
        print("="*50)
        print("중앙 리소스 관리자 초기화를 시작합니다...")

        current_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_dir = os.path.join(current_dir, "dataset")

        self.PATHS = {
            "kobart_model": os.path.join(dataset_dir, "kobart_checkpoint-10000"),
            "kot5_model": os.path.join(dataset_dir, "kot5_checkpoint-146000"),
            "kobart_g2s_model": os.path.join(dataset_dir, "kobart_lemma_to_sentence_model"),
            "source_csv": os.path.join(dataset_dir, "동영상가능데이터셋.txt"), # 원본 데이터셋
            "lemma_txt": os.path.join(dataset_dir, "extract_lemma.txt"),   # 추출해서 만들 파일
            "cache_sbert": os.path.join(dataset_dir, "our_lemma_cache_sbert.pt"),
            "cache_kosim_roberta": os.path.join(dataset_dir, "our_lemma_cache_kosim_roberta.pt"),
            "cache_kosim_multitask": os.path.join(dataset_dir, "our_lemma_cache_kosim_multitask.pt"),
            "cache_bge_m3_korean": os.path.join(dataset_dir, "our_lemma_cache_bge_m3_korean.pt"),
            "cache_ko_sroberta_multitask": os.path.join(dataset_dir, "our_lemma_cache_ko_sroberta_multitask.pt"),
        }
        
        self.OLLAMA_MODEL_NAME = 'exaone3.5:7.8b'

        # --- 모델 로딩 로직 ---
        print("\n[1/5] KoBART & KoT5 모델 로딩...")
        self.kobart_tokenizer = PreTrainedTokenizerFast.from_pretrained(self.PATHS["kobart_model"])
        self.kobart_model = BartForConditionalGeneration.from_pretrained(self.PATHS["kobart_model"])
        self.kot5_tokenizer = T5TokenizerFast.from_pretrained(self.PATHS["kot5_model"])
        self.kot5_model = T5ForConditionalGeneration.from_pretrained(self.PATHS["kot5_model"])
        self.kobart_g2s_tokenizer = PreTrainedTokenizerFast.from_pretrained(self.PATHS["kobart_g2s_model"])
        self.kobart_g2s_model = BartForConditionalGeneration.from_pretrained(self.PATHS["kobart_g2s_model"])

        print("\n[2/5] SBERT 계열 모델 5종 로딩...")
        self.SBERT_MODELS = { "sbert": SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS"), "kosim_roberta": SentenceTransformer("BM-K/KoSimCSE-roberta"), "kosim_multitask": SentenceTransformer("BM-K/KoSimCSE-RoBERTa-multitask"), "bge_m3_korean": SentenceTransformer("upskyy/bge-m3-korean"), "ko_sroberta_multitask": SentenceTransformer("jhgan/ko-sroberta-multitask") }
        
        # --- 단어 사전 로딩 로직 변경 ---
        print("\n[3/5] 단어 사전 로딩...")
        self.lemmas_set = self._initialize_lemma_data() 
        self.lemma_list = sorted(list(self.lemmas_set))

        # --- 사전 임베딩 및 캐시 관리 ---
        print("\n[4/5] 사전 임베딩 및 캐시 관리...")
        self.SBERT_EMBEDDINGS = { "sbert": self._get_or_create_embeddings("sbert", self.PATHS["cache_sbert"]), "kosim_roberta": self._get_or_create_embeddings("kosim_roberta", self.PATHS["cache_kosim_roberta"]), "kosim_multitask": self._get_or_create_embeddings("kosim_multitask", self.PATHS["cache_kosim_multitask"]), "bge_m3_korean": self._get_or_create_embeddings("bge_m3_korean", self.PATHS["cache_bge_m3_korean"]), "ko_sroberta_multitask": self._get_or_create_embeddings("ko_sroberta_multitask", self.PATHS["cache_ko_sroberta_multitask"]), }
        
        print("\n[5/5] Ollama 로컬 모델 확인...")
        self.initialize_ollama_model()
            
        print("\n모든 리소스 로딩 완료!")
        print("="*50)

    def _create_lemma_txt_from_csv(self, csv_path, txt_path):
        """[신규] 원본 CSV를 읽어 'kor' 열을 추출하고 TXT 파일로 저장하는 함수"""
        print(f"원본 CSV '{os.path.basename(csv_path)}'에서 단어 목록을 추출합니다...")
        lemmas_set = set()
        try:
            with open(csv_path, 'r', encoding='utf-8') as f_in:
                reader = csv.reader(f_in)
                next(reader, None)  # 헤더 행 건너뛰기
                for row in reader:
                    if len(row) >= 2:
                        word = row[1].strip()
                        if word:
                            lemmas_set.add(word)
            
            with open(txt_path, 'w', encoding='utf-8') as f_out:
                for word in sorted(list(lemmas_set)):
                    f_out.write(word + '\n')
            
            print(f"'{os.path.basename(txt_path)}' 생성 완료 (총 {len(lemmas_set)}개 단어).")
            return True
        except FileNotFoundError:
            print(f"오류: 원본 CSV 파일 '{csv_path}'를 찾을 수 없습니다!")
            return False
        except Exception as e:
            print(f"오류: CSV 처리 중 에러 발생: {e}")
            return False

    def _initialize_lemma_data(self):
        """[신규] lemma.txt가 없으면 만들고, 있으면 로드하는 통합 함수"""
        source_csv_path = self.PATHS["source_csv"]
        target_txt_path = self.PATHS["lemma_txt"]

        if not os.path.exists(target_txt_path):
            print(f"단어 목록 파일 '{os.path.basename(target_txt_path)}'이 없어 새로 생성합니다.")
            creation_success = self._create_lemma_txt_from_csv(source_csv_path, target_txt_path)
            if not creation_success:
                print("단어 목록 생성 실패. 빈 목록으로 계속합니다.")
                return set()

        lemmas_set = set()
        try:
            with open(target_txt_path, 'r', encoding='utf-8') as f:
                for line in f:
                    lemmas_set.add(line.strip())
            print(f"단어 목록 파일 로딩 성공 (총 {len(lemmas_set)}개 단어).")
        except FileNotFoundError:
            print(f"오류: 파일을 생성했음에도 '{os.path.basename(target_txt_path)}'를 찾을 수 없습니다.")
        return lemmas_set

    def _get_or_create_embeddings(self, model_name, cache_path):
        """(내용 변경 없음)"""
        model = self.SBERT_MODELS[model_name]
        try:
            lemma_list_cache, lemma_embeddings = torch.load(cache_path)
            if lemma_list_cache == self.lemma_list:
                print(f"'{model_name}' 모델 캐시 로딩 성공.")
                return lemma_embeddings
            else:
                print(f"'{model_name}' 모델 캐시 새로 생성 (단어 목록 변경됨)")
        except Exception:
            print(f"'{model_name}' 모델 캐시 없음 또는 오류. 새로 생성합니다.")
        
        lemma_embeddings = model.encode(self.lemma_list, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=True)
        torch.save((self.lemma_list, lemma_embeddings), cache_path)
        return lemma_embeddings

    def initialize_ollama_model(self):
        """
        Ollama 모델이 로컬에 존재하는지 확인하고, 없으면 자동으로 다운로드합니다.
        """
        print(f"\n--- Ollama 모델 '{self.OLLAMA_MODEL_NAME}' 확인 중 ---")
        try:
            installed_models = [m.model for m in ollama.list()['models']]

            base_model_name = self.OLLAMA_MODEL_NAME.split(':')[0]
            model_is_installed = any(name.startswith(base_model_name) for name in installed_models)

            if model_is_installed:
                print(f"'{base_model_name}' 계열의 모델이 이미 설치되어 있습니다. 다운로드를 건너뜁니다.")
                return

            print(f"모델 '{self.OLLAMA_MODEL_NAME}'을(를) 찾을 수 없습니다. 다운로드를 시작합니다...")
            print("이 작업은 모델 크기에 따라 시간이 다소 걸릴 수 있습니다.")
            
            current_status = ""
            for progress in ollama.pull(self.OLLAMA_MODEL_NAME, stream=True):
                status = progress.get('status')
                if status != current_status:
                    current_status = status
                    sys.stdout.write(f'\r  - 상태: {current_status}')
                    sys.stdout.flush()

            print("\n모델 다운로드 완료")

        except Exception as e:
            print(f"\nOllama 모델 초기화 중 오류 발생: {e}")
            print("   Ollama 애플리케이션이 백그라운드에서 실행 중인지 확인해주세요.")
resources = ResourceManager()