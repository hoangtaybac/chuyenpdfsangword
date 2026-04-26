import os, re, tempfile, base64, shutil, subprocess, uuid
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from PyPDF2 import PdfReader

try:
    from mistralai.client import Mistral
except Exception:
    from mistralai import Mistral

app = FastAPI(title="PDF OCR API", version="1.1.0")
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

class ExportDocxPayload(BaseModel):
    content: str = ""
    images: Dict[str, str] = {}
    title: Optional[str] = "ket-qua-ocr"

class ExportPreviewHtmlPayload(BaseModel):
    html: str = ""
    title: Optional[str] = "ket-qua-ocr"

@app.get("/")
def root():
    return {"ok": True, "service": "PDF OCR API", "endpoint": "POST /ocr", "export": "POST /export-docx", "export_preview": "POST /export-docx-preview"}

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


def _safe_filename(name: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9_\-.]+", "-", name or "ket-qua-ocr").strip("-_.")
    return name or "ket-qua-ocr"


def _decode_image_to_file(img_id: str, b64: str, img_dir: Path) -> Optional[Path]:
    try:
        raw = b64.split(",", 1)[1] if b64.startswith("data:") and "," in b64 else b64
        data = base64.b64decode(raw)
        ext = Path(img_id).suffix.lower() or ".jpg"
        if ext not in [".jpg", ".jpeg", ".png", ".gif", ".webp"]:
            ext = ".jpg"
        out = img_dir / (Path(img_id).stem + ext)
        out.write_bytes(data)
        return out
    except Exception:
        return None


def _prepare_markdown_for_docx(content: str, images: Dict[str, str], workdir: Path) -> str:
    """Chuẩn hóa Markdown để Pandoc chuyển $...$/$$...$$ thành Word Equation thật (OMML)."""
    md = content or ""

    # Bỏ vài ký hiệu Markdown thừa do OCR để Word gọn hơn.
    md = re.sub(r"^\s*```.*?$", "", md, flags=re.MULTILINE)
    md = re.sub(r"\*\*\s*(Lời\s*giải\s*:?)\s*\*\*", r"\n\n**\1**\n", md, flags=re.I)

    img_dir = workdir / "images"
    img_dir.mkdir(exist_ok=True)
    for img_id, b64 in (images or {}).items():
        img_path = _decode_image_to_file(img_id, b64, img_dir)
        if not img_path:
            continue
        rel = img_path.relative_to(workdir).as_posix()
        md_img = f"\n\n![{img_id}]({rel})\n\n"
        patterns = [
            r"\[\s*HÌNH\s*:\s*" + re.escape(img_id) + r"\s*\]",
            r"\[\s*Hình\s*:\s*" + re.escape(img_id) + r"\s*\]",
            re.escape(img_id),
        ]
        for pat in patterns:
            md = re.sub(pat, md_img, md, flags=re.I)

    # Giảm lỗi bảng markdown OCR: bỏ hàng chỉ toàn --- nếu bị đứng riêng.
    md = re.sub(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$", "", md, flags=re.MULTILINE)
    return md.strip() + "\n"

@app.post("/export-docx")
async def export_docx(payload: ExportDocxPayload):
    if not payload.content.strip():
        raise HTTPException(status_code=400, detail="Chưa có nội dung để xuất Word")

    pandoc_bin = os.getenv("PANDOC_PATH") or shutil.which("pandoc")
    if not pandoc_bin:
        raise HTTPException(
            status_code=500,
            detail="Server chưa có Pandoc nên chưa xuất được Word Equation thật. Cần cài pandoc hoặc đặt biến PANDOC_PATH."
        )

    tmp_root = Path(tempfile.mkdtemp(prefix="docx_export_"))
    try:
        md = _prepare_markdown_for_docx(payload.content, payload.images, tmp_root)
        md_path = tmp_root / "input.md"
        docx_path = tmp_root / f"{uuid.uuid4().hex}.docx"
        md_path.write_text(md, encoding="utf-8")

        cmd = [
            pandoc_bin,
            str(md_path),
            "-f", "markdown+tex_math_dollars+tex_math_single_backslash+pipe_tables",
            "-t", "docx",
            "--resource-path", str(tmp_root),
            "-o", str(docx_path),
        ]
        completed = subprocess.run(cmd, cwd=str(tmp_root), capture_output=True, text=True, timeout=120)
        if completed.returncode != 0 or not docx_path.exists():
            raise RuntimeError(completed.stderr or completed.stdout or "Pandoc không tạo được file docx")

        filename = _safe_filename(payload.title or "ket-qua-ocr") + ".docx"
        return FileResponse(
            path=str(docx_path),
            filename=filename,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            background=None,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xuất Word: {e}")


def _prepare_preview_html_for_docx(html: str, workdir: Path) -> str:
    """Nhận đúng HTML phần xem trước, tách data:image ra file thật để Pandoc không lặp/không mất ảnh."""
    html = html or ""
    img_dir = workdir / "preview_images"
    img_dir.mkdir(exist_ok=True)

    html = re.sub(r"\[\s*HÌNH\s*:?\s*\]", "", html, flags=re.I)
    html = re.sub(r"^\s*\]\s*$", "", html, flags=re.M)

    def repl_img(match):
        before, src, after = match.group(1), match.group(2), match.group(3)
        if not src.startswith("data:image"):
            return match.group(0)
        try:
            header, raw = src.split(",", 1)
            ext = ".png"
            if "jpeg" in header or "jpg" in header:
                ext = ".jpg"
            elif "gif" in header:
                ext = ".gif"
            elif "webp" in header:
                ext = ".webp"
            data = base64.b64decode(raw)
            out = img_dir / f"img_{uuid.uuid4().hex}{ext}"
            out.write_bytes(data)
            rel = out.relative_to(workdir).as_posix()
            return f'<img{before} src="{rel}"{after}>'
        except Exception:
            return ""

    html = re.sub(r"<img([^>]*?)\s+src=[\"']([^\"']+)[\"']([^>]*)>", repl_img, html, flags=re.I)
    html = re.sub(r"(?im)^\s*img-\d+\.(?:jpe?g|png|webp)\s*$", "", html)

    style = """
    <style>
      body{font-family:Arial, sans-serif;font-size:12pt;line-height:1.25;color:#111827;}
      p{margin:2px 0;}
      img{max-width:55%;display:block;margin:8px auto;border:0;}
      table{border-collapse:collapse;margin:10px 0;}
      td,th{border:1px solid #334155;padding:6px 10px;text-align:center;vertical-align:middle;}
      .solution-title{display:block;width:100%;text-align:center;color:#0f3d91;font-size:13pt;font-weight:700;margin:8px 0 6px;}
    </style>
    """

    if "<html" not in html.lower():
        html = f'<!doctype html><html><head><meta charset="utf-8">{style}</head><body>{html}</body></html>'
    elif "</head>" in html.lower():
        html = re.sub(r"</head>", style + "</head>", html, count=1, flags=re.I)
    return html

@app.post("/export-docx-preview")
async def export_docx_preview(payload: ExportPreviewHtmlPayload):
    if not payload.html.strip():
        raise HTTPException(status_code=400, detail="Chưa có nội dung xem trước để xuất Word")

    pandoc_bin = os.getenv("PANDOC_PATH") or shutil.which("pandoc")
    if not pandoc_bin:
        raise HTTPException(status_code=500, detail="Server chưa có Pandoc nên chưa xuất được Word. Cần cài pandoc hoặc đặt biến PANDOC_PATH.")

    tmp_root = Path(tempfile.mkdtemp(prefix="docx_preview_export_"))
    try:
        html = _prepare_preview_html_for_docx(payload.html, tmp_root)
        html_path = tmp_root / "preview.html"
        docx_path = tmp_root / f"{uuid.uuid4().hex}.docx"
        html_path.write_text(html, encoding="utf-8")
        cmd = [pandoc_bin, str(html_path), "-f", "html+tex_math_dollars+tex_math_single_backslash", "-t", "docx", "--resource-path", str(tmp_root), "-o", str(docx_path)]
        completed = subprocess.run(cmd, cwd=str(tmp_root), capture_output=True, text=True, timeout=120)
        if completed.returncode != 0 or not docx_path.exists():
            raise RuntimeError(completed.stderr or completed.stdout or "Pandoc không tạo được file docx")
        filename = _safe_filename(payload.title or "ket-qua-ocr") + ".docx"
        return FileResponse(path=str(docx_path), filename=filename, media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document", background=None)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xuất Word từ phần xem trước: {e}")

