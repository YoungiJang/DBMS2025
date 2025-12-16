import os
import re
from glob import glob
from langchain_community.document_loaders import PyPDFLoader

def clean_hyphenated_text(text: str) -> str:
    """
    논문 하이픈+엔터(hyphenation) 문제 해결
    예: "algo-\nrithm" -> "algorithm"
    """
    return re.sub(r'(?<=[a-z])-\n(?=[a-z])', '', text)

def data_loader(directory_path):
    #PDF 파일 목록 찾기
    pdf_files = glob(os.path.join(directory_path, "*.pdf"))
    
    if not pdf_files:
        print(f"Error: No PDF files found in '{directory_path}'.")
        return

    print(f"Found {len(pdf_files)} PDF files. Starting lazy loading process...")

    # 파일별로 루프를 돌며 Lazy Loading 수행
    for file_path in pdf_files:
        filename=os.path.relpath(file_path,directory_path)
        print(f"Title:{filename}")
        try:
            #page 단위로 자르면 중간에 잘리게 되는 paragraph 발생, single 모드로 큰 document 하나 전달 후 splitting 과정에서 작은 단위로 자름
            loader = PyPDFLoader(file_path,mode='single')
            for document in loader.lazy_load():
                document.page_content = clean_hyphenated_text(document.page_content)
                yield document  # 한 페이지씩 다음 단계로 전달
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")