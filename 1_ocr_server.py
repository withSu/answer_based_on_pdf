import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import gc
import fitz
from sentence_transformers import SentenceTransformer
import chromadb
import openai
import re
from dotenv import load_dotenv
from PIL import Image, ImageEnhance, ImageOps
import io
import numpy as np
from paddleocr import PaddleOCR
import pytesseract
import easyocr
import time
from transformers import BlipProcessor, BlipForConditionalGeneration
from pykospacing import Spacing
import pandas as pd
from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn

# 모델 초기화
spacing_model = Spacing()
reader_easyocr = easyocr.Reader(['ko', 'en'], gpu=False)

# Tesseract 경로 설정 (macOS와 리눅스에 따라 다름)
if os.path.exists("/opt/homebrew/bin/tesseract"):  # macOS
    pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"
else:  # 리눅스
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# 경로 설정
PDF_DIR = "./pdf_data"
os.makedirs(PDF_DIR, exist_ok=True)

# 모델 로드
ocr = PaddleOCR(use_angle_cls=True, lang="korean", rec_batch_num=1, gpu=False)
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
model = SentenceTransformer("all-MiniLM-L6-v2")
client_db = chromadb.PersistentClient(path="./chroma_db")
collection = client_db.get_or_create_collection("pdf_page_vectors")
DEBUG = True

# FastAPI 앱 생성
app = FastAPI(title="RAG API Server", description="PDF 데이터 기반 RAG 및 OCR 처리 API")

# 기존 함수들
def enhance_image(img):
    img = img.convert("L")
    img = ImageOps.autocontrast(img)
    img = ImageEnhance.Contrast(img).enhance(2.0)
    img = img.resize((img.width * 2, img.height * 2))
    return img

def correct_spacing(text):
    try:
        return spacing_model(text)
    except:
        return text

def extract_tables_as_markdown(image):
    text = pytesseract.image_to_string(image, lang='kor+eng')
    lines = [line for line in text.split("\n") if line.strip()]
    table_lines = [line for line in lines if re.search(r'\s{2,}|\t+', line)]
    if not table_lines:
        return "No obvious table-like structure detected."
    data = [re.split(r'\s{2,}|\t+', line) for line in table_lines]
    try:
        df = pd.DataFrame(data)
        return df.to_markdown(index=False)
    except Exception as e:
        return "Table conversion failed: " + str(e)

def ensemble_ocr(image):
    image = enhance_image(image)
    paddle_result = ocr.ocr(np.array(image), cls=True)
    paddle_text = " ".join(word_info[1][0] for line in paddle_result if line for word_info in line)
    tesseract_text = pytesseract.image_to_string(image, lang='kor+eng')
    easy_text = " ".join([item[1] for item in reader_easyocr.readtext(np.array(image))])
    return {
        "paddle": correct_spacing(paddle_text.strip()),
        "tesseract": correct_spacing(tesseract_text.strip()),
        "easyocr": correct_spacing(easy_text.strip())
    }

def postprocess_ocr_text(text):
    corrections = {"세종대황": "세종대왕"}
    for wrong, correct in corrections.items():
        text = text.replace(wrong, correct)
    return text

def process_page(page, page_num):
    plain_text = page.get_text().strip()
    image_list = page.get_images(full=True)
    ocr_results, captions, tables = [], [], []

    for img_index, img in enumerate(image_list):
        xref = img[0]
        base_image = page.parent.extract_image(xref)
        image = Image.open(io.BytesIO(base_image["image"]))

        ocr_dict = ensemble_ocr(image)
        ocr_text = postprocess_ocr_text(
            f"[PaddleOCR]\n{ocr_dict['paddle']}\n\n[Tesseract]\n{ocr_dict['tesseract']}\n\n[EasyOCR]\n{ocr_dict['easyocr']}")
        ocr_results.append(f"--- 이미지 {img_index + 1} OCR 결과 ---\n{ocr_text}")

        try:
            caption_inputs = blip_processor(image, return_tensors="pt")
            caption_output = blip_model.generate(**caption_inputs)
            caption = blip_processor.decode(caption_output[0], skip_special_tokens=True)
            captions.append(f"--- 이미지 {img_index + 1} 캡션 ---\n{caption}")
        except Exception as e:
            captions.append(f"--- 이미지 {img_index + 1} 캡션 생성 실패: {str(e)}")

        try:
            table_md = extract_tables_as_markdown(image)
            tables.append(f"--- 이미지 {img_index + 1} Table Extraction ---\n{table_md}")
        except Exception as e:
            tables.append(f"--- 이미지 {img_index + 1} Table Extraction 실패: {str(e)}")

    combined_text = (
        f"[Plain Text]\n{plain_text}\n\n"
        f"[Ensemble OCR]\n{chr(10).join(ocr_results)}\n\n"
        f"[Image Captions]\n{chr(10).join(captions)}\n\n"
        f"[Tables]\n{chr(10).join(tables)}"
    )

    if DEBUG:
        print(f"----- Page {page_num + 1} - Combined Text -----")
        print(combined_text)
        print("----- End of Combined Text -----\n")

    embedding = model.encode([combined_text])[0]
    collection.upsert(
        embeddings=[embedding.tolist()],
        documents=[combined_text],
        metadatas=[{"page": page_num + 1}],
        ids=[str(page_num)]
    )
    del plain_text, ocr_results, captions, tables, combined_text, embedding, image_list
    gc.collect()

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    for page_num, page in enumerate(doc):
        process_page(page, page_num)
    doc.close()

def extract_texts_from_multiple_pdfs(pdf_dir):
    total_files = 0
    total_pages = 0
    for filename in os.listdir(pdf_dir):
        if filename.lower().endswith(".pdf"):
            total_files += 1
            pdf_path = os.path.join(pdf_dir, filename)
            print(f"\n--- {filename} 처리 시작 ---")
            doc = fitz.open(pdf_path)
            total_pages += len(doc)
            doc.close()
            extract_text_from_pdf(pdf_path)
            print(f"--- {filename} 처리 완료 ---\n")
    return total_files, total_pages

def answer_question(question, collection, model):
    page_match = re.search(r'(\d+)페이지', question)
    target_page = int(page_match.group(1)) if page_match else None
    question_embedding = model.encode([question])

    if target_page:
        filter_pages = [target_page - 1, target_page, target_page + 1]
        filter_pages = [p for p in filter_pages if p > 0]
        results = collection.query(
            query_embeddings=question_embedding.tolist(),
            n_results=3,
            where={"page": {"$in": filter_pages}}
        )
    else:
        results = collection.query(
            query_embeddings=question_embedding.tolist(),
            n_results=3
        )

    relevant_pages = results.get("documents", [[]])[0]
    page_numbers = [meta["page"] for meta in results.get("metadatas", [[]])[0]]
    context = "\n".join(relevant_pages) if relevant_pages else "관련 정보를 찾을 수 없습니다."

    if context.strip() == "" or context.strip() == "관련 정보를 찾을 수 없습니다.":
        first_answer = "답: 모르겠습니다"
        fallback_prompt = (
            f"위 질문은 PDF 내에서는 관련 정보를 찾지 못했다. "
            f"따라서 일반적인 지식을 활용하여 아래 형식에 맞춰 답변하라.\n\n"
            f"질문: {question}\n\n"
            f"답변 형식은 반드시 다음과 같다. 첫 번째 줄에 '답: 모르겠습니다'를 표기한다. "
            f"두 번째 줄에는 '다만, 제가 생각하기에는'으로 시작하여 추가 설명을 작성한다. "
            f"세 번째 줄에는 '제가 생각한 답 :' 뒤에 최종 정답을 작성한다. "
            f"추가로, 질문에 보기가 전혀 없으면 해당 문제를 OX 문제로 간주하여 'O' 또는 'X'로 답하고, "
            f"보기가 ',' 또는 '/'로 구분되어 있다면 객관식 문제로 판단하여 그에 따른 정답을 선택하라."
        )
        fallback_response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": fallback_prompt}]
        )
        second_answer = fallback_response.choices[0].message.content
        return first_answer + "\n\nGPT기반 추가 답변:\n" + second_answer
    else:
        prompt = (
            f"다음 PDF에서 제공된 컨텍스트를 바탕으로 아래 프롬프트에 대해 답변한다.\n\n"
            f"프롬프트: {question}\n\n"
            f"컨텍스트:\n{context}\n\n"
            f"페이지 번호는 {page_numbers}이다.\n"
            f"입력된 프롬프트가 평서문 형식이더라도 반드시 질문으로 인식하고, 주어진 텍스트에 대한 적절한 답을 도출하라.\n"
            f"만약 프롬프트 끝에 보기가 포함되어 있으면, 보기는 콤마(,)로 구분된 답 후보임을 인지하고 그 중에서 정답을 선택하라. "
            f"보기가 없으면 해당 프롬프트를 OX 문제로 간주하여 'O' 또는 'X'로 답하라.\n"
            f"답변은 반드시 '답: 정답' 형식으로 정답을 먼저 명시한 후, 그에 대한 설명을 덧붙여라.\n"
            f"가능한 한 PDF의 내용에서 추론하며, 관련 정보가 부족할 경우 '모르겠습니다. 다만, 제가 생각하기에는'라는 서술어를 포함하여 답변하라."
        )
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

# API 요청 및 응답 모델
class QuestionRequest(BaseModel):
    question: str
    page: Optional[int] = None

class OCRTextRequest(BaseModel):
    text: str

class OCRImageRequest(BaseModel):
    image_base64: str
    format: str = "png"  # 이미지 형식 (png, jpg 등)

class EmbeddingRequest(BaseModel):
    execute: bool = True

class APIResponse(BaseModel):
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None

# API 엔드포인트
@app.post("/ask", response_model=APIResponse)
async def api_ask_question(request: QuestionRequest):
    """
    RAG 시스템에 질문을 하고 답변을 받습니다.
    """
    try:
        # 페이지 번호가 지정된 경우 질문에 추가
        question = request.question
        if request.page:
            question = f"{question} {request.page}페이지"
        
        # 답변 생성
        start_time = time.time()
        answer = answer_question(question, collection, model)
        elapsed_time = time.time() - start_time
        
        return {
            "status": "success",
            "message": "질문에 대한 답변이 생성되었습니다.",
            "data": {
                "question": question,
                "answer": answer,
                "elapsed_time": f"{elapsed_time:.2f}초"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"질문 처리 중 오류 발생: {str(e)}")

@app.post("/process_ocr_text", response_model=APIResponse)
async def process_ocr_text(request: OCRTextRequest):
    """
    OCR로 추출된 텍스트를 처리하고 답변을 생성합니다.
    """
    try:
        # OCR 텍스트를 질문으로 처리
        question = request.text
        
        # 답변 생성
        start_time = time.time()
        answer = answer_question(question, collection, model)
        elapsed_time = time.time() - start_time
        
        return {
            "status": "success",
            "message": "OCR 텍스트에 대한 답변이 생성되었습니다.",
            "data": {
                "question": question,
                "answer": answer,
                "elapsed_time": f"{elapsed_time:.2f}초"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR 텍스트 처리 중 오류 발생: {str(e)}")

@app.post("/process_ocr_image", response_model=APIResponse)
async def process_ocr_image(request: OCRImageRequest):
    """
    Base64로 인코딩된 이미지를 받아 OCR 처리 후 답변을 생성합니다.
    """
    try:
        import base64
        
        # Base64 디코딩하여 이미지 생성
        image_data = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_data))
        
        # OCR 처리
        ocr_results = ensemble_ocr(image)
        
        # 앙상블 결과 결합
        combined_text = f"{ocr_results['paddle']}\n{ocr_results['tesseract']}\n{ocr_results['easyocr']}"
        
        # 답변 생성
        start_time = time.time()
        answer = answer_question(combined_text, collection, model)
        elapsed_time = time.time() - start_time
        
        return {
            "status": "success",
            "message": "이미지 OCR 처리 및 답변 생성 완료",
            "data": {
                "ocr_text": combined_text,
                "paddle_ocr": ocr_results['paddle'],
                "tesseract_ocr": ocr_results['tesseract'],
                "easyocr_ocr": ocr_results['easyocr'],
                "answer": answer,
                "elapsed_time": f"{elapsed_time:.2f}초"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"이미지 OCR 처리 중 오류 발생: {str(e)}")

@app.post("/embed_pdfs", response_model=APIResponse)
async def embed_pdfs(request: EmbeddingRequest = Body(...)):
    """
    PDF 디렉토리의 모든 PDF 파일을 임베딩합니다.
    """
    if not request.execute:
        return {
            "status": "info",
            "message": "임베딩이 요청되지 않았습니다. 실행하려면 execute=true를 설정하세요.",
            "data": None
        }
    
    try:
        start_time = time.time()
        total_files, total_pages = extract_texts_from_multiple_pdfs(PDF_DIR)
        elapsed_time = time.time() - start_time
        
        return {
            "status": "success",
            "message": "PDF 임베딩이 완료되었습니다.",
            "data": {
                "total_files": total_files,
                "total_pages": total_pages,
                "elapsed_time": f"{elapsed_time:.2f}초",
                "pdf_directory": PDF_DIR
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF 임베딩 중 오류 발생: {str(e)}")

@app.get("/status", response_model=APIResponse)
async def get_status():
    """
    서버 상태와 임베딩된 데이터 정보를 반환합니다.
    """
    try:
        # 현재 임베딩된 데이터 정보 가져오기
        collection_info = collection.count()
        
        return {
            "status": "success",
            "message": "서버가 정상적으로 실행 중입니다.",
            "data": {
                "embedded_documents": collection_info,
                "pdf_directory": PDF_DIR,
                "server_time": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"상태 확인 중 오류 발생: {str(e)}")

# 서버 실행
if __name__ == "__main__":
    print("=======================================")
    print("  RAG API 서버 시작")
    print("  * OCR 및 PDF 데이터 기반 질의응답 API")
    print("=======================================")
    print(f"PDF 디렉토리: {PDF_DIR}")
    print(f"임베딩 컬렉션: {collection.name}")
    print("서버 시작 중...")
    uvicorn.run(app, host="0.0.0.0", port=8000)