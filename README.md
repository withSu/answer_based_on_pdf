

# 📄 DocuRAG

GPT만으로는 해결할 수 없는 PDF 기반 전문 문서를 참고하여, 질문에 대해 정확하고 근거 있는 답변을 생성하는 오픈소스 문서 질의응답 시스템이다.  
대학생들의 전공 공부, 리서치, 과제 수행 시 실질적인 자료 기반 학습을 지원한다.

---

# 🎯 1. 프로젝트 목적

**DocuRAG**는 오픈소스 기술만을 이용해 PDF 문서를 임베딩하고, 질문에 대해 OpenAI GPT를 활용해 답변을 생성하는,  
경량화된 서버-클라이언트 기반 문서 질의응답 시스템이다.

문서 임베딩과 검색은 모두 로컬에서 무료로 수행되며,  
GPT API는 최종 답변 생성에만 사용하여 **비용을 혁신적으로 절감**하였다.

이 프로젝트는 다음을 목표로 한다.

- 문서 임베딩과 검색은 **100% 오픈소스**로 처리하여 **API 비용 0원**
- 질문 답변 생성에만 GPT API를 사용해 **비용 최소화**
- 서버-클라이언트 구조를 통해 **부하를 분산**하고, **확장성을 확보**
- 복합 OCR을 통해 **PDF 내 표, 이미지까지 정확히 인식**

---

# 🚀 2. 주요 특징

## 🧠 2.1 오픈소스 기반 문서 임베딩 및 검색

SentenceTransformer와 ChromaDB를 활용해 문서를 로컬에서 무료로 임베딩하고 검색할 수 있다.

## 🤖 2.2 GPT 기반 자연어 답변 생성

검색된 문서를 기반으로 OpenAI GPT-4 Turbo를 사용하여 고품질 답변을 생성한다.

## 🏗️ 2.3 서버-클라이언트 분리 구조

FastAPI 서버가 모든 부하를 처리하고, 클라이언트는 요청만 수행하여 리소스 소모를 최소화하였다.

## 🖼️ 2.4 복합 OCR 처리 및 표 인식

PaddleOCR, EasyOCR, Tesseract를 조합하여 표, 그림 캡션 등 다양한 형식을 정확하게 텍스트화한다.

---

# 🛠️ 3. 시스템 아키텍처

## 3.1 전체 흐름

```
(1) PDF 문서 업로드
       ↓
(2) 페이지 단위로 텍스트 추출
      └─ (텍스트가 없으면) 이미지에서 OCR 수행
             - PaddleOCR / EasyOCR / Tesseract 사용
             - 표 인식 및 이미지 캡션 추출
       ↓
(3) 추출된 텍스트 통합
      └─ [Plain Text + OCR Text + Table Text + Image Captions]
       ↓
(4) 문서 임베딩
      └─ SentenceTransformer("all-MiniLM-L6-v2")로 임베딩 생성
       ↓
(5) 임베딩 저장
      └─ ChromaDB(Local Persistent) 컬렉션에 저장
```

---

## 3.2 질의응답 처리 흐름

```
(1) 사용자가 질문 입력
       ↓
(2) 질문 임베딩 생성
      └─ SentenceTransformer 사용
       ↓
(3) ChromaDB에서 유사 문서 검색
      └─ (Top-K 가장 유사한 문서 반환)
       ↓
(4) 검색된 문서를 Context로 GPT에게 질문
      └─ OpenAI GPT-4 Turbo API 호출
       ↓
(5) 최종 자연어 답변 생성 및 반환
```

---

## 3.3 기술 스택 상세 매칭

| 단계 | 사용 기술 | 설명 |
|:---|:---|:---|
| 텍스트 추출 | PyMuPDF, PaddleOCR, EasyOCR, Tesseract | PDF 페이지 텍스트 추출, OCR 필요 시 자동 전환 |
| 텍스트 후처리 | PyKoSpacing, Hanspell | 띄어쓰기 및 맞춤법 보정 |
| 임베딩 생성 | SentenceTransformer("all-MiniLM-L6-v2") | 문서 및 질문 임베딩 생성 |
| 벡터 저장/검색 | ChromaDB | 문서 임베딩 저장 및 유사도 검색 |
| 질문 답변 생성 | OpenAI GPT-4 Turbo API | 검색된 문서 컨텍스트 기반 답변 생성 |
| 서버 구축 | FastAPI + Uvicorn | 서버 및 API 통신 처리 |

---

# 📂 4. 디렉토리 구조

```
ocr-capture-system/
│
├── 1_ocr_server.py         # FastAPI 기반 OCR + RAG 서버
├── 2_embed_pdfs.py         # PDF 문서 임베딩 스크립트
├── 3_client.py             # 질의응답 요청용 간단한 클라이언트
├── chroma_db/              # 문서 임베딩 저장소 (ChromaDB)
├── pdf_data/               # 질의응답용 PDF 파일 저장 폴더
├── .env                    # API 키 환경변수 파일
├── environment.yml         # Conda 환경 설정 파일
└── README.md               # 프로젝트 설명 문서
```

---

# 🛠️ 5. 설치 방법

## 🔹 5.1 Conda 환경 구성

```bash
conda env create -f environment.yml
conda activate rag_ocr
```

## 🔹 5.2 환경 변수 설정

`.env` 파일을 생성하고 다음 내용을 추가한다.

```env
OPENAI_API_KEY=your_openai_api_key_here
```

---

# ▶️ 6. 실행 방법

## 🖥️ 6.1 서버 실행

```bash
python 1_ocr_server.py
```

## 📄 6.2 PDF 임베딩

```bash
python 2_embed_pdfs.py
```

## ❓ 6.3 질문 요청 (클라이언트)

```bash
python 3_client.py
```

---

# 📡 7. API 명세

- `POST /ask` : 질문을 전송하고 GPT 기반 답변을 받는다.
- `POST /process_ocr_text` : OCR 텍스트로 질의응답을 수행한다.
- `POST /process_ocr_image` : Base64 이미지로 OCR 및 질문-답변을 수행한다.
- `POST /embed_pdfs` : PDF 전체를 임베딩한다.
- `GET /status` : 서버 상태 및 문서 수를 확인한다.

---

# ⚡ 8. 요약

- 문서 임베딩 및 검색은 **전적으로 오픈소스** 기반
- GPT API는 **최종 답변 생성에만 사용**하여 비용 절감
- 서버-클라이언트 구조로 **확장성과 효율성**을 확보
- 복합 OCR 처리로 **다양한 문서 구조 지원**

---

# 🪪 9. 라이선스

MIT License

---

# 🔥 최종 정리

**DocuRAG**는  
"오픈소스 문서 검색"과  
"GPT 기반 자연어 답변"을 결합하여,  
**최소 비용으로 대규모 문서 기반 학습과 연구를 지원하는 최적화된 오픈소스 솔루션**이다.

---