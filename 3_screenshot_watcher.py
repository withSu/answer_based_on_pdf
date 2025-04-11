import tkinter as tk
import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import requests
import base64
import subprocess

# 스크린샷이 저장되는 기본 폴더 (macOS)
SCREENSHOT_DIR = os.path.expanduser("~/Desktop")
API_URL = "http://localhost:8000/process_ocr_image"

class ScreenshotHandler(FileSystemEventHandler):
    def __init__(self, api_url, status_callback=None):
        self.api_url = api_url
        self.last_processed = ""
        self.status_callback = status_callback
    
    def on_created(self, event):
        if event.is_directory:
            return
            
        # 숨김 파일 및 임시 파일 무시
        filename = os.path.basename(event.src_path)
        if filename.startswith('.'):
            return
            
        if event.src_path.endswith(('.png', '.jpg', '.jpeg')):
            # 파일이 완전히 저장될 때까지 약간 대기
            time.sleep(0.5)
            
            # 파일이 실제로 존재하는지 확인
            if os.path.exists(event.src_path) and os.path.getsize(event.src_path) > 0:
                print(f"새 스크린샷 감지: {event.src_path}")
                if self.status_callback:
                    self.status_callback(f"스크린샷 감지: {os.path.basename(event.src_path)}")
                self.process_image(event.src_path)
    
    def on_moved(self, event):
        if event.is_directory:
            return
        
        # 숨김 파일에서 실제 파일로 이름 변경되는 경우 처리
        if event.dest_path.endswith(('.png', '.jpg', '.jpeg')):
            # 파일이 완전히 저장될 때까지 약간 대기
            time.sleep(0.5)
            
            if os.path.exists(event.dest_path) and os.path.getsize(event.dest_path) > 0:
                print(f"새 스크린샷 감지(이름 변경): {event.dest_path}")
                if self.status_callback:
                    self.status_callback(f"스크린샷 감지: {os.path.basename(event.dest_path)}")
                self.process_image(event.dest_path)
    
    def process_image(self, image_path):
        # 파일이 존재하는지 한번 더 확인
        if not os.path.exists(image_path):
            print(f"파일이 존재하지 않음: {image_path}")
            if self.status_callback:
                self.status_callback("오류: 파일이 존재하지 않습니다")
            return
            
        # 중복 처리 방지
        if image_path == self.last_processed:
            return
        self.last_processed = image_path
        
        # 이미지 처리 및 서버로 전송
        try:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode()
            
            print("OCR 서버로 이미지 전송 중...")
            if self.status_callback:
                self.status_callback("OCR 서버로 이미지 전송 중...")
            
            response = requests.post(
                self.api_url,
                json={"image_base64": encoded_string, "format": "png"},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # OCR 결과 표시
                ocr_text = f"PaddleOCR: {result['data'].get('paddle_ocr', '결과 없음')}\n\n"
                ocr_text += f"Tesseract: {result['data'].get('tesseract_ocr', '결과 없음')}\n\n"
                ocr_text += f"EasyOCR: {result['data'].get('easyocr_ocr', '결과 없음')}"
                
                print("\n===== OCR 결과 =====")
                print(ocr_text)
                
                print("\n===== 분석 결과 =====")
                answer = result["data"]["answer"]
                print(answer)
                
                if self.status_callback:
                    self.status_callback(f"OCR 처리 완료 ({result['data'].get('elapsed_time', 'N/A')})")
                
                # 결과 알림 표시 (macOS)
                try:
                    subprocess.run(["osascript", "-e", f'display notification "{answer}" with title "OCR 결과"'])
                except:
                    pass
                
                return ocr_text, answer
            else:
                error_msg = f"API 오류: {response.status_code}"
                print(error_msg)
                if self.status_callback:
                    self.status_callback(f"오류: {error_msg}")
                return None, None
        except Exception as e:
            error_msg = f"이미지 처리 중 오류: {str(e)}"
            print(error_msg)
            if self.status_callback:
                self.status_callback(f"오류: {str(e)}")
            return None, None

class OCRCaptureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RAG OCR 캡처 앱")
        self.root.geometry("800x600")
        
        # UI 요소 생성
        self.create_widgets()
        
        # 스크린샷 감시 시작
        self.event_handler = ScreenshotHandler(API_URL, self.update_status)
        self.observer = Observer()
        self.observer.schedule(self.event_handler, path=SCREENSHOT_DIR, recursive=False)
        self.observer.start()
        
        # 오버레이 창
        self.overlay = None
        
        # 종료 시 정리
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def create_widgets(self):
        # 상단 프레임
        top_frame = tk.Frame(self.root, padx=10, pady=10)
        top_frame.pack(fill=tk.X)
        
        tk.Label(top_frame, text="RAG OCR 캡처 앱", font=("Arial", 16, "bold")).pack()
        tk.Label(top_frame, text="스크린샷을 찍으면 자동으로 OCR 처리하여 RAG 시스템에 질의합니다").pack(pady=5)
        
        # 버튼 프레임
        button_frame = tk.Frame(self.root, padx=10, pady=5)
        button_frame.pack(fill=tk.X)
        
        self.overlay_button = tk.Button(button_frame, text="투명 오버레이 표시", command=self.show_overlay)
        self.overlay_button.pack(side=tk.LEFT, padx=5)
        
        self.check_button = tk.Button(button_frame, text="서버 상태 확인", command=self.check_server_status)
        self.check_button.pack(side=tk.LEFT, padx=5)
        
        self.clear_button = tk.Button(button_frame, text="화면 초기화", command=self.clear_display)
        self.clear_button.pack(side=tk.LEFT, padx=5)
        
        # 구분선
        separator = tk.Frame(self.root, height=2, bd=1, relief=tk.SUNKEN)
        separator.pack(fill=tk.X, padx=5, pady=5)
        
        # 콘텐츠 프레임
        content_frame = tk.Frame(self.root, padx=10, pady=5)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # OCR 텍스트 영역
        ocr_frame = tk.LabelFrame(content_frame, text="OCR 텍스트", padx=5, pady=5)
        ocr_frame.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        
        self.ocr_text = tk.Text(ocr_frame, wrap=tk.WORD, width=40, height=20)
        self.ocr_text.pack(fill=tk.BOTH, expand=True)
        
        # 결과 영역
        result_frame = tk.LabelFrame(content_frame, text="분석 결과", padx=5, pady=5)
        result_frame.pack(fill=tk.BOTH, expand=True, side=tk.RIGHT)
        
        self.result_text = tk.Text(result_frame, wrap=tk.WORD, width=40, height=20)
        self.result_text.pack(fill=tk.BOTH, expand=True)
        
        # 상태바
        self.status_bar = tk.Label(self.root, text="준비 완료", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def show_overlay(self):
        if self.overlay and self.overlay.winfo_exists():
            self.overlay.destroy()
            self.overlay_button.config(text="투명 오버레이 표시")
            return
            
        # 오버레이 창 생성
        self.overlay = tk.Toplevel(self.root)
        self.overlay.attributes('-fullscreen', True)
        self.overlay.attributes('-alpha', 0.2)  # 투명도 설정
        self.overlay.configure(bg='gray')
        
        # 안내 레이블
        label = tk.Label(
            self.overlay, 
            text="Shift+Command+4로 영역을 선택하세요\nESC 키로 종료",
            bg='black', fg='white', font=('Arial', 14),
            padx=10, pady=10
        )
        label.pack(pady=50)
        
        # ESC 키로 종료
        self.overlay.bind("<Escape>", lambda e: self.hide_overlay())
        
        # 버튼 텍스트 변경
        self.overlay_button.config(text="오버레이 숨기기")
    
    def hide_overlay(self):
        if self.overlay and self.overlay.winfo_exists():
            self.overlay.destroy()
            self.overlay_button.config(text="투명 오버레이 표시")
    
    def check_server_status(self):
        try:
            self.update_status("서버 상태 확인 중...")
            response = requests.get(f"{API_URL.replace('/process_ocr_image', '')}/status", timeout=5)
            
            if response.status_code == 200:
                result = response.json()
                
                self.clear_display()
                self.result_text.insert(tk.END, "서버 상태 정보:\n\n")
                
                if 'data' in result:
                    data = result['data']
                    self.result_text.insert(tk.END, f"임베딩된 문서: {data.get('embedded_documents', '정보 없음')}\n")
                    self.result_text.insert(tk.END, f"PDF 디렉토리: {data.get('pdf_directory', '정보 없음')}\n")
                    self.result_text.insert(tk.END, f"서버 시간: {data.get('server_time', '정보 없음')}\n")
                
                self.update_status("서버 상태 확인 완료")
            else:
                self.update_status(f"서버 상태 확인 실패: {response.status_code}")
        except Exception as e:
            self.update_status(f"서버 연결 오류: {str(e)}")
    
    def clear_display(self):
        self.ocr_text.delete(1.0, tk.END)
        self.result_text.delete(1.0, tk.END)
    
    def update_status(self, message):
        self.status_bar.config(text=message)
        self.root.update()
        print(message)
    
    def display_ocr_results(self, ocr_text, answer):
        if ocr_text:
            self.ocr_text.delete(1.0, tk.END)
            self.ocr_text.insert(tk.END, ocr_text)
        
        if answer:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, answer)
    
    def on_closing(self):
        if self.overlay and self.overlay.winfo_exists():
            self.overlay.destroy()
        
        self.observer.stop()
        self.observer.join()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = OCRCaptureApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()