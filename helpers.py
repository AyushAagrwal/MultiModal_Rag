import os
import io
import json
import uuid
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Tuple

# OpenAI
from dotenv import load_dotenv
from openai import OpenAI

# Vector DB
import faiss

# PDF / OCR
import fitz  # PyMuPDF
import pytesseract
from PIL import Image

# ---- Load env + client
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---- Tesseract path (adjust if needed, keep if already set)
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\ayush.agarwal\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# -----------------------------
# CONFIG
# -----------------------------
EMBED_MODEL = "text-embedding-3-small"   # 1536 dims
GEN_MODEL   = "gpt-4o-mini"

INDEX_DIR = "index_data"

TEXT_INDEX_FILE = os.path.join(INDEX_DIR, "faiss_text.index")
TEXT_META_FILE  = os.path.join(INDEX_DIR, "metadata_text.jsonl")

IMAGE_INDEX_FILE = os.path.join(INDEX_DIR, "faiss_image.index")
IMAGE_META_FILE  = os.path.join(INDEX_DIR, "metadata_image.jsonl")

PDF_SAVE_PATH = os.path.join(INDEX_DIR, "uploaded.pdf")
IMAGES_DIR    = os.path.join(INDEX_DIR, "images")
STATUS_FILE   = os.path.join(INDEX_DIR, "upload_status.txt")

# Track latest upload for “latest-only” retrieval
LATEST_DOC_FILE = os.path.join(INDEX_DIR, "latest_doc_id.txt")

TOP_K_DEFAULT      = 1
TOP_K_EXPLORATORY  = 5
CHUNK_SIZE         = 800
CHUNK_OVERLAP      = 200
TEXT_MINLEN_FOR_NO_OCR = 30

os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# -----------------------------
# UTILITIES
# -----------------------------
def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"

def normalize(vectors: np.ndarray) -> np.ndarray:
    if vectors.size == 0:
        return vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    return vectors / norms

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Tuple[str, int, int]]:
    text = text or ""
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        chunks.append((chunk, start, end))
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks

def embed_texts(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1536), dtype="float32")
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = [d.embedding for d in resp.data]
    return np.array(vecs, dtype="float32")

def ensure_index(path: str, dim: int = 1536) -> faiss.Index:
    if os.path.exists(path):
        return faiss.read_index(path)
    return faiss.IndexFlatIP(dim)

def save_index(index: faiss.Index, path: str):
    faiss.write_index(index, path)

def append_metadata(path: str, records: List[Dict[str, Any]]):
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def load_metadata(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def clear_all_indexes():
    for p in [TEXT_INDEX_FILE, TEXT_META_FILE, IMAGE_INDEX_FILE, IMAGE_META_FILE, PDF_SAVE_PATH, STATUS_FILE, LATEST_DOC_FILE]:
        if os.path.exists(p):
            os.remove(p)
    # Clean images folder
    for fname in os.listdir(IMAGES_DIR):
        try:
            os.remove(os.path.join(IMAGES_DIR, fname))
        except Exception:
            pass

# -----------------------------
# OCR / PDF helpers
# -----------------------------
def ocr_image(pil_image: Image.Image) -> str:
    try:
        return pytesseract.image_to_string(pil_image) or ""
    except Exception:
        return ""

def extract_page_text_or_ocr(doc: fitz.Document, page_num: int) -> str:
    page = doc.load_page(page_num)
    text = page.get_text("text") or ""
    if len(text.strip()) >= TEXT_MINLEN_FOR_NO_OCR:
        return text
    try:
        pix = page.get_pixmap(dpi=200)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        ocr_text = ocr_image(img)
        return (text + "\n" + ocr_text).strip()
    except Exception:
        return text

def extract_images_from_pdf(doc: fitz.Document, source_name: str) -> List[Dict[str, Any]]:
    out = []
    for pno in range(len(doc)):
        page = doc.load_page(pno)
        images = page.get_images(full=True)
        for _, img in enumerate(images):
            xref = img[0]
            try:
                base = doc.extract_image(xref)
                img_bytes = base["image"]
                ext = base.get("ext", "png")
                if ext.lower() not in ["png", "jpg", "jpeg"]:
                    ext = "png"
                img_id = str(uuid.uuid4())
                filename = f"{img_id}.{ext}"
                save_path = os.path.join(IMAGES_DIR, filename)
                with open(save_path, "wb") as f:
                    f.write(img_bytes)

                pil_img = Image.open(io.BytesIO(img_bytes))
                caption_text = ocr_image(pil_img).strip()
                if not caption_text:
                    caption_text = f"Image extracted from {source_name} page {pno+1}"

                out.append({
                    "id": img_id,
                    "source": source_name,
                    "page": pno + 1,
                    "file_type": "image",
                    "uploaded_at": now_iso(),
                    "image_path": f"/images/{filename}",
                    "caption": caption_text
                })
            except Exception:
                continue
    return out

# -----------------------------
# Latest-doc helpers (for “latest-only” retrieval)
# -----------------------------
def set_current_doc_id(doc_id: str):
    with open(LATEST_DOC_FILE, "w", encoding="utf-8") as f:
        f.write(doc_id or "")

def get_current_doc_id():
    if not os.path.exists(LATEST_DOC_FILE):
        return None
    try:
        with open(LATEST_DOC_FILE, "r", encoding="utf-8") as f:
            val = (f.read() or "").strip()
            return val or None
    except Exception:
        return None
