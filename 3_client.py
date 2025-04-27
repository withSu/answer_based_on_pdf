#!/usr/bin/env python3
# ocr_hotkey_client.py

import os
import time
import threading
import tempfile
import subprocess
import requests
import numpy as np
from PIL import Image
import easyocr
import tkinter as tk
from tkinter import ttk, font
from pynput import keyboard

# ==== 설정 =====================================================
API_URL = "http://192.168.0.35:8000/process_ocr_text"  # FastAPI 서버 주소
SCREENSHOT_CMD = "screencapture"                       # macOS 스크린샷 명령어
# ==============================================================

class OCRCaptureApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RAG OCR Capture (Drag & Hotkey)")
        self.geometry("600x500")
        self.reader = None
        self.capture_area = None
        self.current_keys = set()
        self.capturing = False

        # 핫키 설정
        self.capture_hotkey = {keyboard.Key.ctrl, keyboard.Key.shift}  # Ctrl+Shift: 드래그→OCR→전송
        self.send_hotkey    = {keyboard.Key.ctrl, keyboard.Key.alt}    # Ctrl+Option: 현재 OCR만 전송

        # UI 생성
        self._build_main_ui()
        self._build_response_window()
        self._load_model_async()
        self._register_hotkeys()

    def _build_main_ui(self):
        """메인 윈도우 UI"""
        frm = tk.Frame(self, padx=10, pady=10)
        frm.pack(fill=tk.BOTH, expand=True)

        self.progress = ttk.Progressbar(frm, mode="indeterminate")
        self.progress.pack(fill=tk.X, pady=(0,10))

        tk.Label(
            frm,
            text="Ctrl+Shift: drag → OCR → send\nCtrl+Option: send current OCR only",
            justify="left"
        ).pack(pady=(0,20))

        tk.Label(frm, text="OCR Text:").pack(anchor="w")
        self.ocr_text = tk.Text(frm, height=6)
        self.ocr_text.pack(fill=tk.BOTH, pady=5)

        tk.Button(frm, text="Send to API", command=self.send_to_api).pack(pady=(0,10))

        tk.Label(frm, text="Selected Area:").pack(anchor="w")
        self.area_label = tk.Label(frm, text="(none)", fg="gray")
        self.area_label.pack(anchor="w", pady=(0,10))

        self.status_bar = tk.Label(
            self, text="Loading OCR model...", bd=1, relief=tk.SUNKEN, anchor=tk.W
        )
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    def _build_response_window(self):
        """API 응답을 표시할 별도 창 생성"""
        self.resp_win = tk.Toplevel(self)
        self.resp_win.title("API Response")
        # 닫히지 않도록 최소화만 가능
        self.resp_win.protocol("WM_DELETE_WINDOW", lambda: None)
        
        # 응답창 상단에 상태 표시 레이블 추가
        # 노란색 배경의 상태 표시 레이블을 오른쪽 위에 배치
        state_frame = tk.Frame(self.resp_win)
        state_frame.pack(fill=tk.X, side=tk.TOP)
        
        # 상태 표시 레이블 (노란색 배경, 검정색 글씨, 오른쪽 정렬)
        self.resp_state_label = tk.Label(
            state_frame, 
            text="READY", 
            bg="yellow", 
            fg="black",  # 검정색 글씨로 명시적 설정
            font=("Arial", 15),
            padx=5,
            pady=2
        )
        self.resp_state_label.pack(side=tk.RIGHT, anchor=tk.NE)

        # 글씨 크기를 5배로 키운 폰트
        large_font = font.Font(family="Arial", size=13)
        self.resp_text = tk.Text(
            self.resp_win, font=large_font, wrap=tk.WORD, state="disabled"
        )
        self.resp_text.pack(fill=tk.BOTH, expand=True)

    def _load_model_async(self):
        """EasyOCR 모델 로드 비동기"""
        self.progress.start(10)
        threading.Thread(target=self._load_model, daemon=True).start()

    def _load_model(self):
        self.reader = easyocr.Reader(['ko','en'], gpu=False)
        self.after(0, self._on_model_loaded)

    def _on_model_loaded(self):
        self.progress.stop()
        self.progress.pack_forget()
        self.status("READY")

    def status(self, msg: str):
        """상태바 업데이트"""
        self.status_bar.config(text=msg)
        # 응답창의 상태 레이블도 함께 업데이트
        self.resp_state_label.config(text=msg)
        self.update_idletasks()

    def _register_hotkeys(self):
        """키보드 리스너 등록"""
        listener = keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release
        )
        listener.daemon = True
        listener.start()

    def _on_key_press(self, key):
        self.current_keys.add(key)
        # Ctrl+Shift: drag → OCR → send
        if self.capture_hotkey.issubset(self.current_keys) and not self.capturing:
            threading.Thread(target=self.capture_and_ocr, daemon=True).start()
        # Ctrl+Option: send only
        if self.send_hotkey.issubset(self.current_keys):
            self.send_to_api()

    def _on_key_release(self, key):
        self.current_keys.discard(key)

    def capture_and_ocr(self):
        """드래그 영역 선택 → 캡처 → OCR → 전송"""
        self.capturing = True
        self.status("Select area…")
        time.sleep(0.1)  # 키 해제 시간

        # 투명 오버레이로 영역 선택
        x, y, w, h = self._select_area_transparent()
        if w == 0 or h == 0:
            self.status("Cancelled")
            self.capturing = False
            return

        self.area_label.config(text=f"{x},{y},{w},{h}")

        # macOS screencapture 로만 영역 저장
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.close()
        subprocess.run([
            SCREENSHOT_CMD, "-x",
            "-R", f"{x},{y},{w},{h}",
            tmp.name
        ], check=True)

        # OCR
        try:
            img = Image.open(tmp.name).convert("RGB")
            text = " ".join(r[1] for r in self.reader.readtext(np.array(img)))
        except Exception as e:
            text = f"[ERROR reading image] {e}"
        finally:
            os.unlink(tmp.name)

        # 메인 창에 OCR 텍스트 삽입
        self.ocr_text.delete("1.0", tk.END)
        self.ocr_text.insert(tk.END, text)

        # API 전송
        self.send_to_api()
        self.capturing = False

    def _select_area_transparent(self):
        """
        듀얼 모니터 전 화면에 완전 투명 오버레이를 띄워
        드래그한 영역의 (x,y,w,h)를 반환.
        """
        sel = {'x':0,'y':0,'w':0,'h':0}
        done = tk.BooleanVar(value=False)
        overlays = []

        # 각 모니터마다 투명 윈도우 생성
        try:
            from screeninfo import get_monitors
            mons = get_monitors()
        except ImportError:
            mons = [self.winfo_screenwidth()]

        for mon in mons:
            # 모니터 정보가 객체이면 width, height, x, y 사용
            mx, my = getattr(mon, 'x', 0), getattr(mon, 'y', 0)
            mw = getattr(mon, 'width', self.winfo_screenwidth())
            mh = getattr(mon, 'height', self.winfo_screenheight())

            ov = tk.Toplevel(self)
            ov.overrideredirect(True)
            ov.attributes("-topmost", True)
            ov.attributes("-fullscreen", False)
            ov.attributes("-alpha", 0.0)  # 완전 투명
            ov.geometry(f"{mw}x{mh}+{mx}+{my}")

            cv = tk.Canvas(ov, cursor="cross", highlightthickness=0)
            cv.pack(fill=tk.BOTH, expand=True)

            def _press(e, ox=mx, oy=my):
                sel['x'], sel['y'] = e.x + ox, e.y + oy

            def _release(e, ox=mx, oy=my):
                x2, y2 = e.x + ox, e.y + oy
                sel['w'], sel['h'] = abs(x2 - sel['x']), abs(y2 - sel['y'])
                sel['x'], sel['y'] = min(sel['x'], x2), min(sel['y'], y2)
                done.set(True)

            cv.bind("<ButtonPress-1>",  _press)
            cv.bind("<ButtonRelease-1>", _release)
            ov.bind("<Escape>",           lambda e: done.set(True))

            overlays.append(ov)

        # 선택 완료 신호 대기
        self.wait_variable(done)

        # 모든 오버레이 제거
        for ov in overlays:
            try: ov.destroy()
            except: pass

        return sel['x'], sel['y'], sel['w'], sel['h']

    def send_to_api(self):
        """현재 OCR 텍스트를 API에 전송하고, 응답을 별도 창에 표시"""
        txt = self.ocr_text.get("1.0", tk.END).strip()
        if not txt:
            return
        self.status("Sending to API…")
        try:
            resp = requests.post(API_URL, json={"text": txt}, timeout=30)
            data = resp.json()
            ans = data.get("data", {}).get("answer", resp.text)
        except Exception as e:
            ans = f"Error: {e}"

        # 응답 창에 표시
        self.resp_text.config(state="normal")
        self.resp_text.delete("1.0", tk.END)
        self.resp_text.insert(tk.END, ans)
        self.resp_text.config(state="disabled")
        self.status("Done")

if __name__ == "__main__":
    OCRCaptureApp().mainloop()