import requests
import time
import os
import sys

def embed_pdfs(api_url="http://localhost:8000"):
    """PDF 임베딩을 시작합니다."""
    
    print("=======================================")
    print("  PDF 임베딩 시작")
    print("=======================================")
    
    # PDF 디렉토리 확인
    pdf_dir = "./pdf_data"
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir)
        print(f"PDF 디렉토리 '{pdf_dir}'가 생성되었습니다.")
        print("이 디렉토리에 임베딩할 PDF 파일을 넣고 다시 실행하세요.")
        return False
    
    # PDF 파일 확인
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"PDF 디렉토리 '{pdf_dir}'에 PDF 파일이 없습니다.")
        print("임베딩할 PDF 파일을 넣고 다시 실행하세요.")
        return False
    
    print(f"총 {len(pdf_files)}개의 PDF 파일이 발견되었습니다:")
    for i, file in enumerate(pdf_files, 1):
        print(f"  {i}. {file}")
    
    # 임베딩 API 호출
    print("\n임베딩을 시작합니다. 이 과정은 시간이 오래 걸릴 수 있습니다...")
    try:
        start_time = time.time()
        response = requests.post(
            f"{api_url}/embed_pdfs",
            json={"execute": True},
        )
        
        if response.status_code == 200:
            result = response.json()
            total_time = time.time() - start_time
            
            print("\n임베딩이 완료되었습니다!")
            print(f"총 파일 수: {result['data'].get('total_files', '정보 없음')}")
            print(f"총 페이지 수: {result['data'].get('total_pages', '정보 없음')}")
            print(f"처리 시간: {result['data'].get('elapsed_time', f'{total_time:.2f}초')}")
            return True
        else:
            print(f"\n오류가 발생했습니다 (HTTP {response.status_code}):")
            print(response.text)
            return False
    except requests.RequestException as e:
        print(f"\n연결 오류: {str(e)}")
        print("서버가 실행 중인지 확인하세요.")
        return False
    except KeyboardInterrupt:
        print("\n임베딩이 사용자에 의해 중단되었습니다.")
        return False

if __name__ == "__main__":
    # 명령줄 인수로 API URL을 받을 수 있음
    api_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    embed_pdfs(api_url)