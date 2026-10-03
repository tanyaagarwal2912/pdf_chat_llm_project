"""
PDF question answering (basic RAG), questions typed in the console

Steps:
1) Read the PDF and split the text into chunks
2) Embed every chunk (cached on disk so it happens only once)
3) The console asks you for a question
4) Retrieve the most similar chunks and let Gemini answer from them

Install:  pip install google-genai pypdf numpy
Run:      python pdf_rag.py
"""

import json
import logging
import os
import time

import numpy as np
from google import genai
from pypdf import PdfReader

logging.getLogger("google_genai").setLevel(logging.ERROR)

# ---------------- settings ----------------
PDF_PATH = "your_file.pdf"                 # <- change to your PDF path
EMBED_MODEL = "gemini-embedding-2"         # same model for chunks and question
GEN_MODEL = "gemini-3.6-flash"
CHUNK_SIZE = 1000                          # characters per chunk
CHUNK_OVERLAP = 200                        # characters shared between neighbouring chunks
TOP_K = 3                                  # how many chunks to send to the model
MIN_SCORE = 0.30                           # below this, treat the question as unrelated (tune it)
# ------------------------------------------

client = genai.Client()


# 1) READ + CHUNK -----------------------------------------------------------
def read_pdf(path):
    """Read a real PDF page by page. If the file is plain text (no %PDF- header), read it as text."""
    with open(path, "rb") as f:
        is_pdf = f.read(5) == b"%PDF-"

    pages = []

    if is_pdf:
        reader = PdfReader(path)
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append((page_number, text))
    else:
        print("Not a real PDF, reading it as plain text.")
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        if text.strip():
            pages.append((1, text))

    return pages


def split_text(text, size, overlap):
    """Split text into chunks of about `size` characters, preferring to cut at sentence ends."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = max(text.rfind(". ", start, end), text.rfind("\n", start, end))
            if cut > start + size // 2:
                end = cut + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def build_chunks(path):
    chunks = []
    for page_number, text in read_pdf(path):
        for piece in split_text(text, CHUNK_SIZE, CHUNK_OVERLAP):
            chunks.append({"page": page_number, "text": piece})
    return chunks


# 2) EMBED ------------------------------------------------------------------
def embed(text):
    for attempt in range(5):
        try:
            result = client.models.embed_content(model=EMBED_MODEL, contents=text)
            return result.embeddings[0].values
        except Exception as error:
            wait = 2 ** attempt
            print(f"  embed failed ({error}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError("Embedding failed after several retries")


def load_or_build_index(path):
    cache_file = os.path.splitext(path)[0] + "_index.json"

    if os.path.exists(cache_file) and os.path.getmtime(cache_file) > os.path.getmtime(path):
        print(f"Loading saved embeddings from {cache_file}")
        with open(cache_file, "r") as f:
            data = json.load(f)
        return data["chunks"], np.array(data["vectors"])

    chunks = build_chunks(path)
    if not chunks:
        raise SystemExit("No text found. The PDF may be scanned (images only) and would need OCR.")

    print(f"Embedding {len(chunks)} chunks (first run only)...")
    vectors = []
    for i, chunk in enumerate(chunks, start=1):
        vectors.append(embed(chunk["text"]))
        if i % 10 == 0 or i == len(chunks):
            print(f"  {i}/{len(chunks)}")

    with open(cache_file, "w") as f:
        json.dump({"chunks": chunks, "vectors": [list(map(float, v)) for v in vectors]}, f)
    print(f"Saved embeddings to {cache_file}")
    return chunks, np.array(vectors)


# 3) RETRIEVE ---------------------------------------------------------------
def retrieve(question, chunks, vectors):
    q = np.array(embed(question))
    scores = vectors @ q / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(q))
    top = np.argsort(scores)[::-1][:TOP_K]
    return [(chunks[i], float(scores[i])) for i in top]


# 4) ANSWER -----------------------------------------------------------------
def answer(question, hits):
    context = "\n\n".join(f"[Page {c['page']}]\n{c['text']}" for c, _ in hits)
    prompt = f"""
Answer the question using only the context below.
If the answer is not in the context, say "I don't know based on the document."

Context:
{context}

Question:
{question}
"""
    response = client.models.generate_content(model=GEN_MODEL, contents=prompt)
    return response.text


def main():
    chunks, vectors = load_or_build_index(PDF_PATH)
    print("\nReady. Ask a question about the document (type 'exit' to quit).")

    while True:
        question = input("\nQuestion: ").strip()

        if question.lower() in ("exit", "quit", "q"):
            print("Bye!")
            break
        if not question:
            continue

        hits = retrieve(question, chunks, vectors)

        print("\nRetrieved context:")
        for chunk, score in hits:
            print(f"  page {chunk['page']}  score {score:.3f}")

        if hits[0][1] < MIN_SCORE:
            print("\nFinal Answer:\nI don't know based on the document.")
            continue

        print("\nFinal Answer:")
        print(answer(question, hits))


if __name__ == "__main__":
    main()