# 🧠 Multimodal RAG System

This project implements a **Retrieval-Augmented Generation (RAG)** system that can process and query **multiple data formats** — text files, images (PNG/JPG), and PDFs (text, images, or both). It uses **OpenAI embeddings**, **FAISS**, and **Flask** to create a lightweight multimodal knowledge retrieval API and web app.

---

## 🚀 Features

### ✅ Data Ingestion and Storage
- Handles `.pdf`, `.png`, `.jpg`, `.jpeg`, `.txt`, and `.docx`
- Extracts:
  - Text directly from files
  - Text from images and scanned PDFs via **OCR (Tesseract)**
- Stores all embeddings in **FAISS vector database**
- Maintains document metadata (type, upload time, page, etc.)

### ✅ Query Handling
- Supports:
  - Specific factual questions
  - Vague/exploratory questions ("summarize", "overview")
  - Cross-modal questions ("find chart showing sales")
- Automatically routes query type:
  - Text → Text embeddings
  - Image/table queries → Image embeddings
  - Exploratory → Both
- Generates contextual answers with **source citations**

### ✅ PDF Processing
- Handles:
  - Text-only PDFs
  - Image-only PDFs (via OCR)
  - Mixed PDFs (text + embedded images)
- Maintains relationship between text and extracted images

### ✅ API Endpoints
| Endpoint | Method | Description |
|-----------|---------|-------------|
| `/upload` | POST | Upload a document (PDF/Text/Image) and index it |
| `/query` | POST | Query across indexed documents |
| `/upload_status` | GET | Check if upload is complete |
| `/images/<filename>` | GET | Serve extracted OCR images |

---

## 🧩 Architecture Overview

```plaintext
          ┌─────────────┐
          │  User Query │
          └──────┬──────┘
                 │
          ┌──────▼──────┐
          │ Flask API   │
          └──────┬──────┘
                 │
      ┌──────────┼──────────┐
      │           │          │
┌─────▼────┐ ┌────▼────┐ ┌───▼────┐
│ PDF OCR  │ │ Text     │ │ Images │
│ (fitz +  │ │ Chunking │ │ OCR/LLM│
│ pytess.) │ │ Embedding│ │ Caption│
└─────┬────┘ └────┬────┘ └───┬────┘
      │            │          │
      └────────────┼──────────┘
                   │
              ┌────▼────┐
              │  FAISS  │
              └────┬────┘
                   │
           ┌───────▼────────┐
           │ GPT-4o Answer   │
           │ Generation +    │
           │ Source Citation │
           └────────────────┘
