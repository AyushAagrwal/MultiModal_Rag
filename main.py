import os
import uuid
from flask import Flask, request, jsonify, render_template, send_from_directory

from helpers import (
    INDEX_DIR, IMAGES_DIR, STATUS_FILE,
    TEXT_INDEX_FILE, IMAGE_INDEX_FILE,
    clear_all_indexes, set_current_doc_id,
)
from rag import (
    build_text_index_from_pdf, build_image_index_from_pdf,
    build_index_from_textfile, add_standalone_image_to_index,
    search_router, generate_answer
)

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    # Clean landing flag (UI enablement only)
    if os.path.exists(STATUS_FILE):
        os.remove(STATUS_FILE)
    return render_template("index.html", answer=None, results=None)

@app.route("/images/<path:filename>")
def serve_image(filename):
    return send_from_directory(IMAGES_DIR, filename)

@app.route("/upload", methods=["POST"])
def upload_unified():
    """
    Upload any of: .pdf, .png/.jpg/.jpeg, .txt/.docx
    Appends to existing FAISS + metadata (persistent KB).
    Also sets 'latest document' for default latest-only answers.
    """
    file = request.files.get("file")
    if not file:
        return "No file uploaded.", 400

    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()

    os.makedirs(INDEX_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)

    document_id = str(uuid.uuid4())
    set_current_doc_id(document_id)  # 🔖 prefer this upload for answers

    success, msg = False, ""

    try:
        if ext == ".pdf":
            pdf_save = os.path.join(INDEX_DIR, filename)
            file.save(pdf_save)
            text_ok = build_text_index_from_pdf(pdf_save, document_id=document_id)
            image_ok = build_image_index_from_pdf(pdf_save, document_id=document_id)
            success = text_ok or image_ok
            msg = f"PDF '{filename}' indexed successfully!" if success else "No data found in PDF."

        elif ext in [".png", ".jpg", ".jpeg"]:
            success = add_standalone_image_to_index(file, document_id=document_id)
            msg = f"Image '{filename}' indexed successfully!" if success else "Image indexing failed."

        elif ext in [".txt", ".docx"]:
            success, result_msg = build_index_from_textfile(file, document_id=document_id)
            msg = f"Text file '{filename}' indexed successfully!" if success else result_msg

        else:
            msg = f"Unsupported file type: {ext}"

    except Exception as e:
        msg = f"Error while processing {filename}: {e}"

    if not success:
        return render_template("index.html", answer=f"⚠️ {msg}", results=None)

    # Flag for UI to enable the query form
    with open(STATUS_FILE, "w") as f:
        f.write("ready")

    success_html = f"""
    <p style='color:green;font-weight:bold;margin:20px 0;'>
      ✅ {msg} You can now ask a question.
    </p>
    """
    return render_template("index.html", answer=None, results=None) + success_html

@app.route("/upload_status")
def upload_status():
    ready = os.path.exists(STATUS_FILE)
    return jsonify({"ready": ready})

@app.route("/query", methods=["POST"])
def query():
    q = (request.form.get("q") or (request.is_json and (request.json.get("q")))) or ""
    q = q.strip()
    if not q:
        return "Missing 'q' parameter.", 400

    # latest-only unless user ticks "Search across all uploads"
    all_docs = (request.form.get("all_docs") == "1") if not request.is_json else bool(request.json.get("all_docs", False))
    latest_only = not all_docs

    if not (os.path.exists(TEXT_INDEX_FILE) or os.path.exists(IMAGE_INDEX_FILE)):
        return render_template("index.html", answer="⚠️ Please upload a document first.", results=None)

    mode, results = search_router(q, latest_only=latest_only)

    try:
        answer = generate_answer(q, results)
    except Exception as e:
        answer = f"(Answer generation unavailable) Error: {e}"

    if request.is_json:
        return jsonify({"mode": mode, "answer": answer, "results": results})
    return render_template("index.html", answer=answer, results=results)

# Optional manual reset
@app.route("/reset", methods=["POST"])
def reset():
    clear_all_indexes()
    return render_template("index.html", answer="🧹 All indexes and metadata cleared. Start fresh!", results=None)

if __name__ == "__main__":
    print("🚀 Starting Multimodal RAG app (persistent indexes, latest-only by default)...")
    os.makedirs(INDEX_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)
        # Use Railway's port dynamically
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, debug=False)
