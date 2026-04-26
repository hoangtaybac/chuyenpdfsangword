import os, re, tempfile, base64, xml.etree.ElementTree as ET, requests
from typing import Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PyPDF2 import PdfReader
try:
    from mistralai.client import Mistral
except Exception:
    from mistralai import Mistral

app = FastAPI(title="PDF OCR API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Khuyến nghị: đặt MISTRAL_API_KEY trong biến môi trường trên Railway/Render.
def get_api_key() -> str:
    key = os.getenv("MISTRAL_API_KEY") or os.getenv("MISTRAL_API_KEY_SECRET")
    if not key:
        raise RuntimeError("Chưa cấu hình biến môi trường MISTRAL_API_KEY trên server")
    return key

def extract_from_dict(data_dict: Dict[str, Any], result_data: Dict[str, Any]):
    for page in data_dict.get("pages", []):
        if page.get("markdown"):
            result_data["text"] += page["markdown"] + "\n\n"
        elif page.get("text"):
            result_data["text"] += page["text"] + "\n\n"
        for img in page.get("images") or []:
            if img.get("id") and img.get("image_base64"):
                result_data["images"][img["id"]] = img["image_base64"]

def extract_with_regex(text: str, result_data: Dict[str, Any]):
    markdown_blocks = re.findall(r'"markdown":\s*"(.*?)"(?=,|\})', text, re.DOTALL)
    for block in markdown_blocks:
        result_data["text"] += block.replace('\\n', '\n').replace('\\"', '"') + "\n\n"
    image_matches = re.findall(r'"id":\s*"(img-\d+\.jpeg)".*?"image_base64":\s*"([^"]+)"', text, re.DOTALL)
    for img_id, img_base64 in image_matches:
        result_data["images"][img_id] = img_base64

def clean_text_and_images(text: str, images: Dict[str, str]) -> str:
    cleaned = text
    cleaned = re.sub(r'OCRPageObject\(.*?\)', '', cleaned)
    cleaned = re.sub(r'OCRPageDimensions\(.*?\)', '', cleaned)
    cleaned = re.sub(r'images=\[\]', '', cleaned)
    cleaned = re.sub(r'index=\d+', '', cleaned)
    cleaned = re.sub(r'(Câu\s+\d+\.?[:]?)', r'\n\n\1', cleaned)
    cleaned = re.sub(r'(Bài\s+\d+\.?[:]?)', r'\n\n\1', cleaned)
    cleaned = re.sub(r'([A-D]\.)', r'\n\1', cleaned)
    cleaned = re.sub(r'!\[\[HÌNH: (img-\d+\.jpeg)\]\]\(\[HÌNH: \1\]\)', r'[HÌNH: \1]', cleaned)
    cleaned = re.sub(r'!\[(img-\d+\.jpeg)\]\(\1\)', r'[HÌNH: \1]', cleaned)
    for img_id in images.keys():
        cleaned = re.sub(r'!\[.*?\]\(.*?' + re.escape(img_id) + r'.*?\)', f'[HÌNH: {img_id}]', cleaned)
        cleaned = re.sub(r'!{1,2}\[' + re.escape(img_id) + r'\]', f'[HÌNH: {img_id}]', cleaned)
        cleaned = re.sub(r'(?<![a-zA-Z0-9\-\.])' + re.escape(img_id) + r'(?![a-zA-Z0-9\-\.])', f'[HÌNH: {img_id}]', cleaned)
        cleaned = re.sub(r'^\s*' + re.escape(img_id) + r'\s*$', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'^\s*HÌNH:\s*.*$', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'^\s*img-\d+\.jpeg\s*$', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'^\s*\]\s*$', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()

@app.get("/")
def root():
    return {"ok": True, "service": "PDF OCR API", "endpoint": "POST /ocr"}

@app.post("/ocr")
async def ocr_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Chỉ nhận file PDF")
    content = await file.read()
    if len(content) > int(os.getenv("MAX_UPLOAD_BYTES", "52428800")):
        raise HTTPException(status_code=413, detail="File quá lớn")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        page_count = len(PdfReader(tmp_path).pages)
        max_pages = int(os.getenv("MAX_PAGES", "100"))
        if page_count > max_pages:
            raise HTTPException(status_code=400, detail=f"PDF có {page_count} trang, vượt giới hạn {max_pages} trang")

        client = Mistral(api_key=get_api_key())
        if not hasattr(client, "ocr"):
            raise RuntimeError("SDK mistralai quá cũ, cần bản có client.ocr")

        uploaded_pdf = client.files.upload(file={"file_name": file.filename, "content": open(tmp_path, "rb")}, purpose="ocr")
        signed_url = client.files.get_signed_url(file_id=uploaded_pdf.id)
        ocr_response = client.ocr.process(
            model=os.getenv("MISTRAL_OCR_MODEL", "mistral-ocr-latest"),
            document={"type": "document_url", "document_url": signed_url.url},
            include_image_base64=True,
        )

        result = {"text": "", "images": {}, "raw_response_size": len(str(ocr_response)), "pages": page_count}
        if hasattr(ocr_response, "model_dump"):
            extract_from_dict(ocr_response.model_dump(), result)
        if not result["text"] and hasattr(ocr_response, "pages"):
            for page in ocr_response.pages:
                result["text"] += (getattr(page, "markdown", None) or getattr(page, "text", "") or "") + "\n\n"
                for img in getattr(page, "images", []) or []:
                    if getattr(img, "id", None) and getattr(img, "image_base64", None):
                        result["images"][img.id] = img.image_base64
        if not result["text"] or not result["images"]:
            extract_with_regex(str(ocr_response), result)
        result["cleaned_text"] = clean_text_and_images(result["text"], result["images"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
