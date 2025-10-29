import os
import io
import uuid
from typing import List, Dict, Any, Tuple

import faiss
import fitz  # PyMuPDF
from PIL import Image
from docx import Document

from helpers import (
    # config
    INDEX_DIR, IMAGES_DIR,
    TEXT_INDEX_FILE, TEXT_META_FILE, IMAGE_INDEX_FILE, IMAGE_META_FILE,
    TOP_K_DEFAULT, TOP_K_EXPLORATORY, CHUNK_SIZE, CHUNK_OVERLAP,
    GEN_MODEL,
    # utils
    now_iso, normalize, chunk_text, embed_texts,
    ensure_index, save_index, append_metadata, load_metadata,
    extract_page_text_or_ocr, extract_images_from_pdf, ocr_image,
    get_current_doc_id,
)

# -----------------------------
# Index builders
# -----------------------------
def build_text_index_from_pdf(pdf_path: str, document_id: str = None):
    doc = fitz.open(pdf_path)
    source_name = os.path.basename(pdf_path)
    document_id = document_id or str(uuid.uuid4())

    all_chunks, metas = [], []

    for p in range(len(doc)):
        page_text = extract_page_text_or_ocr(doc, p)
        page_chunks = chunk_text(page_text, CHUNK_SIZE, CHUNK_OVERLAP)
        for chunk, start, end in page_chunks:
            chunk = (chunk or "").strip()
            if not chunk:
                continue
            metas.append({
                "id": str(uuid.uuid4()),
                "document_id": document_id,
                "document_name": source_name,
                "modality": "text",
                "source": source_name,
                "page": p + 1,
                "char_start": start,
                "char_end": end,
                "file_type": "pdf_text",
                "uploaded_at": now_iso(),
                "text": chunk
            })
            all_chunks.append(chunk)

    if not all_chunks:
        return False

    vecs = normalize(embed_texts(all_chunks))
    index = ensure_index(TEXT_INDEX_FILE, dim=vecs.shape[1])
    index.add(vecs)
    save_index(index, TEXT_INDEX_FILE)
    append_metadata(TEXT_META_FILE, metas)
    return True

def build_image_index_from_pdf(pdf_path: str, document_id: str = None):
    doc = fitz.open(pdf_path)
    source_name = os.path.basename(pdf_path)
    document_id = document_id or str(uuid.uuid4())

    image_metas = extract_images_from_pdf(doc, source_name)
    if not image_metas:
        return False

    for m in image_metas:
        m["document_id"] = document_id
        m["document_name"] = source_name
    captions = [m["caption"] for m in image_metas]

    vecs = normalize(embed_texts(captions))
    index = ensure_index(IMAGE_INDEX_FILE, dim=vecs.shape[1])
    index.add(vecs)
    save_index(index, IMAGE_INDEX_FILE)
    append_metadata(IMAGE_META_FILE, image_metas)
    return True

def build_index_from_textfile(file_storage, document_id: str = None):
    filename = file_storage.filename
    ext = os.path.splitext(filename)[1].lower()
    text_content = ""
    document_id = document_id or str(uuid.uuid4())

    try:
        if ext == ".txt":
            text_content = file_storage.read().decode("utf-8", errors="ignore")
        elif ext == ".docx":
            doc = Document(io.BytesIO(file_storage.read()))
            text_content = "\n".join([p.text for p in doc.paragraphs])
        else:
            return False, f"Unsupported text file type: {ext}"
    except Exception as e:
        return False, f"Error reading file: {e}"

    if not text_content.strip():
        return False, "No readable text content found."

    chunks = chunk_text(text_content, CHUNK_SIZE, CHUNK_OVERLAP)
    all_texts, metas = [], []
    for chunk, start, end in chunks:
        chunk = (chunk or "").strip()
        if not chunk:
            continue
        metas.append({
            "id": str(uuid.uuid4()),
            "document_id": document_id,
            "document_name": filename,
            "modality": "text",
            "source": filename,
            "page": None,
            "char_start": start,
            "char_end": end,
            "file_type": "text",
            "uploaded_at": now_iso(),
            "text": chunk
        })
        all_texts.append(chunk)

    if not all_texts:
        return False, "No valid text chunks to embed."

    vecs = normalize(embed_texts(all_texts))
    index = ensure_index(TEXT_INDEX_FILE, dim=vecs.shape[1])
    index.add(vecs)
    save_index(index, TEXT_INDEX_FILE)
    append_metadata(TEXT_META_FILE, metas)
    return True, "Text file indexed successfully!"

def add_standalone_image_to_index(file_storage, document_id: str = None):
    document_id = document_id or str(uuid.uuid4())

    raw = file_storage.read()
    pil = Image.open(io.BytesIO(raw)).convert("RGB")
    img_id = str(uuid.uuid4())
    filename = f"{img_id}.png"
    save_path = os.path.join(IMAGES_DIR, filename)
    pil.save(save_path, format="PNG")

    caption = ocr_image(pil).strip() or f"Uploaded image ({filename}), no OCR text detected"
    meta = {
        "id": img_id,
        "document_id": document_id,
        "document_name": file_storage.filename or filename,
        "modality": "image",
        "source": file_storage.filename or "uploaded_image",
        "page": None,
        "file_type": "image",
        "uploaded_at": now_iso(),
        "image_path": f"/images/{filename}",
        "caption": caption
    }

    vec = normalize(embed_texts([caption]))
    if vec.size == 0:
        print("❌ No embedding generated for image caption:", caption)
        return False

    index = ensure_index(IMAGE_INDEX_FILE, dim=vec.shape[1])
    index.add(vec)
    save_index(index, IMAGE_INDEX_FILE)
    append_metadata(IMAGE_META_FILE, [meta])
    return True

# -----------------------------
# Retrieval (supports latest-only filtering)
# -----------------------------
def _search_index(index_path: str, meta_path: str, query: str, k: int, filter_doc_id: str = None) -> List[Dict[str, Any]]:
    metas = load_metadata(meta_path)
    if not os.path.exists(index_path) or not metas:
        return []

    index = faiss.read_index(index_path)

    # Overfetch so we can filter by document_id and still return k
    over_k = max(k * 10, k + 10)
    qvec = normalize(embed_texts([query]))
    scores, idxs = index.search(qvec, over_k)
    idxs = idxs[0].tolist()
    scores = scores[0].tolist()

    out = []
    for i, sc in zip(idxs, scores):
        if i < 0 or i >= len(metas):
            continue
        m = metas[i]

        if filter_doc_id and m.get("document_id") != filter_doc_id:
            continue

        text_preview = (m.get("caption") if m.get("modality") == "image" else m.get("text", ""))[:600]
        out.append({
            "rank": len(out) + 1,
            "score": float(sc),
            "modality": m.get("modality", "text"),
            "source": m.get("source"),
            "page": m.get("page"),
            "text": text_preview,
            "image_path": m.get("image_path"),
            "id": m.get("id")
        })
        if len(out) >= k:
            break
    return out

def detect_query_mode(q: str) -> Tuple[str, int]:
    ql = q.lower()
    exploratory_terms = ["summary", "summarize", "overview", "key findings", "high level", "what's in", "what is in", "overall"]
    if any(term in ql for term in exploratory_terms):
        return ("exploratory", TOP_K_EXPLORATORY)
    image_terms = ["chart", "diagram", "figure", "image", "photo", "graph", "plot", "table", "screenshot", "picture", "visual"]
    if any(term in ql for term in image_terms):
        return ("image", TOP_K_DEFAULT)
    return ("text", TOP_K_DEFAULT)

def search_router(query: str, latest_only: bool = True) -> Tuple[str, List[Dict[str, Any]]]:
    mode, k = detect_query_mode(query)
    filter_doc_id = get_current_doc_id() if latest_only else None

    if mode == "text" and os.path.exists(TEXT_INDEX_FILE):
        return (mode, _search_index(TEXT_INDEX_FILE, TEXT_META_FILE, query, k, filter_doc_id))

    if mode == "image" and os.path.exists(IMAGE_INDEX_FILE):
        return (mode, _search_index(IMAGE_INDEX_FILE, IMAGE_META_FILE, query, k, filter_doc_id))

    if not os.path.exists(TEXT_INDEX_FILE) and os.path.exists(IMAGE_INDEX_FILE):
        return ("image", _search_index(IMAGE_INDEX_FILE, IMAGE_META_FILE, query, k, filter_doc_id))

    text_res = _search_index(TEXT_INDEX_FILE, TEXT_META_FILE, query, TOP_K_EXPLORATORY, filter_doc_id)
    img_res  = _search_index(IMAGE_INDEX_FILE, IMAGE_META_FILE, query, TOP_K_EXPLORATORY, filter_doc_id)
    merged = sorted(text_res + img_res, key=lambda r: r["score"], reverse=True)[:TOP_K_EXPLORATORY]
    return ("exploratory", merged)

# -----------------------------
# RAG answer generation
# -----------------------------
def generate_answer(query: str, contexts: List[Dict[str, Any]]) -> str:
    if not contexts:
        return "I don't know based on the indexed documents. Please upload a document or try a different query."

    ctx_lines = []
    for c in contexts:
        modality_tag = "[IMAGE]" if c["modality"] == "image" else "[TEXT]"
        page_str = f" p.{c['page']}" if c.get("page") else ""
        tag = f"{modality_tag} (source: {c['source']}{page_str}, score={c['score']:.3f})"
        snippet = c["text"]
        ctx_lines.append(f"{tag}\n{snippet}\n")
    ctx_block = "\n\n".join(ctx_lines)

    from helpers import client
    system = (
        "You are an intelligent multimodal retrieval assistant. "
        "You have access to retrieved text, OCR outputs, and image captions extracted from PDF pages or standalone images. "
        "Your goal is to provide accurate, concise, and well-grounded answers ONLY using the provided context. "
        "Do not invent facts, speculate, or hallucinate. "
        "If the required information is not clearly present in the context, respond with: "
        "'I don't know based on the indexed documents.'\n\n"

        "Use the following rules while answering:\n"
        "1. Combine insights from both text and image captions when relevant.\n"
        "2. If the context includes multiple sources, merge related information and avoid redundancy.\n"
        "3. When referencing information from the context, always cite the filename and page number as "
        "(source: FILENAME p.PAGE).\n"
        "4. Keep answers factual, crisp, and domain-neutral — avoid opinions or filler phrases.\n"
        "5. If the query is vague (e.g., 'summarize', 'overview'), generate a short, meaningful summary of the retrieved context.\n"
        "6. If the context includes images or charts, describe what they represent in simple language using the image caption text.\n"
        "7. Never mention that you are using OCR or embeddings — just present the information naturally."
    )
    user = f"Question: {query}\n\nContext:\n{ctx_block}"

    resp = client.chat.completions.create(
        model=GEN_MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()
