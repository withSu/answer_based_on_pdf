# answer_based_on_pdf

---

# 📷 OCR Capture System

RAG 기반 PDF 질의응답과 스크린샷 OCR 처리를 통합한 올인원 솔루션

---

## 🔍 소개

**OCR Capture System**은 사용자가 찍은 스크린샷을 자동으로 인식하여 텍스트를 추출하고, PDF 기반 문서 임베딩을 통해 **질문에 대한 실시간 답변**을 생성하는 시스템이다.  
PaddleOCR, EasyOCR, Tesseract를 활용한 앙상블 OCR 결과를 기반으로, SentenceTransformer 임베딩 + GPT 기반 질의응답을 제공한다.

---

## 🧠 주요 기능

- ⌨️ **단축키 기반 스크린샷 OCR 처리** (macOS 기준)
- 🖼️ **스크린샷 자동 감지 및 실시간 처리**
- 📄 **PDF 문서 자동 임베딩 및 질의응답**
- 🤖 **OpenAI GPT 기반 답변 생성**
- 📊 **표 인식 및 이미지 캡션 생성 지원**
- 🌐 **FastAPI 기반 RESTful API 제공**

---

## 🗂️ 디렉토리 구조

```
ocr-capture-system/
│
├── main.py                   # 콘솔 기반 임베딩 + 질의응답 클라이언트
├── server.py                 # FastAPI 기반 RAG + OCR 서버
├── embed_pdf.py              # PDF 자동 임베딩 유틸
├── watch_screenshot.py       # 스크린샷 자동 감지 및 OCR 처리 앱
├── chroma_db/                # ChromaDB 임베딩 저장소
├── pdf_data/                 # 질의응답용 PDF 저장 폴더
├── .env                      # API 키 환경설정
└── README.md                 # 프로젝트 설명
```

---

## 🚀 설치 방법

```bash
# 가상환경 권장
conda create -n rag_ocr python=3.9
conda activate rag_ocr

# 필수 패키지 설치
pip install -r requirements.txt
```

`.env` 파일에 다음 정보를 입력:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

---

## ▶️ 실행 방법

### 1. 서버 실행

```bash
python server.py
```

서버는 `http://localhost:8000`에서 실행됨.

---

### 2. PDF 임베딩 실행

```bash
python embed_pdf.py
```

`./pdf_data` 폴더에 PDF를 넣은 후 실행.

---

### 3. 스크린샷 감지 + OCR + GPT 답변

```bash
python watch_screenshot.py
```

스크린샷 찍으면 자동 OCR → 질문 전송 → 응답 출력됨. (macOS는 `shift + cmd + 4` 권장)

---

## 🧪 API 명세

- `POST /process_ocr_image`: base64 이미지 OCR 처리 및 답변
- `POST /process_ocr_text`: OCR 텍스트 질문 처리
- `POST /embed_pdfs`: PDF 전체 임베딩
- `GET /status`: 서버 상태 및 문서 수 확인

---

## 🛠 기술 스택

- OCR: PaddleOCR, EasyOCR, Tesseract
- 임베딩: SentenceTransformer (MiniLM)
- 질의응답: OpenAI GPT-4
- 서버: FastAPI + Uvicorn
- DB: ChromaDB

---

## 🤝 기여 방법

1. 이 레포지토리를 Fork하세요
2. 브랜치를 생성하세요 (`feature/your-feature`)
3. 수정 후 PR을 요청해주세요

---

## 📄 라이선스

MIT License

---

