"""
Chat with your document (basic RAG) - Streamlit app

Anyone can upload a PDF or a text file and ask questions about it.

Install:  pip install -r requirements.txt
Run:      streamlit run app.py

HOW IT WORKS (the big picture)
------------------------------
RAG = Retrieval Augmented Generation. Instead of sending the whole document to
the AI, we do this:

  1. READ     - extract the text from the uploaded file
  2. CHUNK    - cut the text into small overlapping pieces (chunks)
  3. EMBED    - turn every chunk into a list of numbers (a "vector") that
                represents its meaning
  4. RETRIEVE - turn the question into a vector too, then find the chunks whose
                vectors are most similar (cosine similarity)
  5. ANSWER   - send only those best chunks + the question to Gemini and ask it
                to answer using only that text
"""

# ----------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------
import hashlib   # makes a unique fingerprint of a file (to detect a new upload)
import io        # lets us treat raw bytes like a file (PdfReader needs a file)
import logging   # used to silence the SDK's noisy AFC warning
import os        # used to read the API key from environment variables
import time      # used to wait between retries

import numpy as np                 # maths on vectors (similarity scores)
import streamlit as st             # builds the web page (upload box, chat...)
from google import genai           # Gemini SDK (embeddings + answers)
from pypdf import PdfReader        # reads text out of PDF files

# Hide the harmless "AFC" notice the Gemini SDK prints (we saw it earlier).
# Real errors are still shown.
logging.getLogger("google_genai").setLevel(logging.ERROR)

# ----------------------------------------------------------------------------
# Settings - the values you are most likely to want to change
# ----------------------------------------------------------------------------
EMBED_MODEL = "gemini-embedding-2"   # model that turns text into vectors (use the SAME one for chunks and questions)
GEN_MODEL = "gemini-3.6-flash"       # model that writes the final answer
CHUNK_SIZE = 1000                    # characters per chunk (bigger = more context, less precise)
CHUNK_OVERLAP = 200                  # characters shared by neighbouring chunks, so a sentence cut in half still appears whole somewhere
DEFAULT_TOP_K = 3                    # how many best chunks are sent to the model per question
DEFAULT_MIN_SCORE = 0.30             # if the best chunk scores below this, answer "I don't know" (adjustable in the sidebar)
MAX_CHUNKS = 600                     # safety limit so huge files don't use up your API quota

# If an API error message contains one of these, it is a temporary problem
# (rate limit / server busy), so it is worth trying again after a short wait.
RETRYABLE = ("429", "RESOURCE_EXHAUSTED", "500", "503", "UNAVAILABLE", "DEADLINE")


# ============================================================================
# 1) READ + CHUNK
# ============================================================================
def is_pdf_bytes(data):
    """A real PDF file always starts with the 5 bytes '%PDF-'."""
    return data[:5] == b"%PDF-"


def extract_pages(data):
    """
    Return a list of (page_number, text) from the uploaded file's raw bytes.
    - Real PDF  -> read page by page.
    - Anything else (txt, md, or a file wrongly named .pdf) -> read as plain text.
    """
    pages = []

    if is_pdf_bytes(data):
        # io.BytesIO wraps the bytes so PdfReader can treat them like a file
        reader = PdfReader(io.BytesIO(data))

        # Some PDFs are locked. Try the empty password; if that fails, stop.
        if reader.is_encrypted and not reader.decrypt(""):
            raise ValueError("This PDF is password protected.")

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""     # scanned pages give "" (no text)
            if text.strip():                     # skip empty pages
                pages.append((page_number, text))
    else:
        # errors="ignore" skips characters that can't be decoded instead of crashing
        text = data.decode("utf-8", errors="ignore")
        if text.strip():
            pages.append((1, text))              # plain text has no pages, so call it page 1

    return pages


def split_text(text, size, overlap):
    """Cut one long text into chunks of about `size` characters.
    It tries to cut at the end of a sentence or line so chunks read naturally."""
    chunks = []
    start = 0                                    # where the current chunk begins

    while start < len(text):
        end = min(start + size, len(text))       # where the chunk would end

        # If we are not at the very end, look backwards for a sentence end (". ")
        # or a line break so we don't cut a sentence in half.
        if end < len(text):
            cut = max(text.rfind(". ", start, end), text.rfind("\n", start, end))
            if cut > start + size // 2:          # only use it if it isn't too early
                end = cut + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):                     # reached the end of the text
            break

        # Next chunk starts a little BEFORE this one ended (the overlap).
        # max(..., start + 1) guarantees we always move forward (no infinite loop).
        start = max(end - overlap, start + 1)

    return chunks


def build_chunks(pages, is_pdf):
    """Chunk every page and remember where each chunk came from (its label)."""
    chunks = []
    for page_number, text in pages:
        for piece in split_text(text, CHUNK_SIZE, CHUNK_OVERLAP):
            chunks.append({"page": page_number, "text": piece})

    # Label used in "Sources used": "Page 3" for PDFs, "Part 3" for text files
    for i, chunk in enumerate(chunks, start=1):
        chunk["label"] = f"Page {chunk['page']}" if is_pdf else f"Part {i}"
    return chunks


# ============================================================================
# 2) EMBED  (text -> vector of numbers)
# ============================================================================
def embed(client, text):
    """Ask Gemini for the vector of one piece of text. Retries on temporary errors."""
    last_error = None
    for attempt in range(5):                     # try up to 5 times
        try:
            result = client.models.embed_content(model=EMBED_MODEL, contents=text)
            return result.embeddings[0].values   # the list of numbers
        except Exception as error:
            last_error = error
            # Wrong API key, wrong model name, etc. -> retrying won't help, so fail now
            if not any(code in str(error) for code in RETRYABLE):
                raise
            time.sleep(2 ** attempt)             # wait 1s, 2s, 4s, 8s, 16s (backoff)
    raise RuntimeError(f"Embedding failed after several retries: {last_error}")


def build_index(client, chunks):
    """Embed every chunk, showing a progress bar. Returns a 2D array: one row per chunk."""
    progress = st.progress(0.0, text="Embedding the document...")
    vectors = []
    for i, chunk in enumerate(chunks, start=1):
        vectors.append(embed(client, chunk["text"]))
        progress.progress(i / len(chunks), text=f"Embedding the document... {i}/{len(chunks)}")
    progress.empty()                             # remove the bar when finished
    return np.array(vectors, dtype=float)


# ============================================================================
# 3) RETRIEVE  (find the chunks most similar to the question)
# ============================================================================
def retrieve(client, question, chunks, vectors, top_k):
    """Return the top_k best (chunk, score) pairs for the question."""
    q = np.array(embed(client, question), dtype=float)   # question -> vector

    # Cosine similarity between the question and EVERY chunk at once:
    #   dot product / (length of chunk vector * length of question vector)
    # Result is a number per chunk; closer to 1 = more similar in meaning.
    norms = np.linalg.norm(vectors, axis=1) * np.linalg.norm(q)
    scores = vectors @ q / np.where(norms == 0, 1, norms)    # np.where avoids dividing by zero

    # argsort sorts lowest->highest, [::-1] flips it, [:top_k] keeps the best few
    top = np.argsort(scores)[::-1][:top_k]
    return [(chunks[i], float(scores[i])) for i in top]


# ============================================================================
# 4) ANSWER  (let Gemini write the answer from the retrieved chunks)
# ============================================================================
def answer(client, question, hits):
    """Send the best chunks + the question to Gemini and return its answer."""
    # Join the chunks into one block of text, each tagged with its page/part
    context = "\n\n".join(f"[{c['label']}]\n{c['text']}" for c, _ in hits)

    # The instructions stop Gemini from using outside knowledge or guessing
    prompt = f"""
Answer the question using only the context below.
If the answer is not in the context, say "I don't know based on the document."

Context:
{context}

Question:
{question}
"""
    response = client.models.generate_content(model=GEN_MODEL, contents=prompt)
    return response.text or "I couldn't generate an answer. Please try rephrasing the question."


# ============================================================================
# UI  (everything the user sees in the browser)
# ============================================================================
def get_client(typed_key):
    """Create the Gemini client. Key order: sidebar box > Streamlit secret > terminal variable."""
    key = typed_key.strip() if typed_key else ""

    if not key:
        try:
            key = st.secrets.get("GEMINI_API_KEY", "")   # used when deployed online
        except Exception:
            key = ""                                     # no secrets file: that's fine

    if not key:
        # This is the one you set with: export GEMINI_API_KEY="..."
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""

    return genai.Client(api_key=key) if key else None


def show_sources(sources):
    """Show an expandable box listing which chunks the answer was based on."""
    if not sources:
        return
    with st.expander("Sources used"):
        for s in sources:
            st.markdown(f"**{s['label']}**  (match score {s['score']:.2f})")
            st.text(s["text"][:500] + ("..." if len(s["text"]) > 500 else ""))


def main():
    # IMPORTANT Streamlit idea: every time you click or type, Streamlit re-runs
    # this whole function from top to bottom. Anything that must survive between
    # runs (chat history, embeddings) is stored in st.session_state.
    st.set_page_config(page_title="Chat with your document", page_icon="📄")
    st.title("📄 Chat with your document")
    st.caption("Upload a PDF or a text file, then ask questions about it.")

    # First run only: create an empty chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ---- Sidebar: settings -------------------------------------------------
    with st.sidebar:
        st.header("Settings")
        typed_key = st.text_input(
            "Gemini API key",
            type="password",                 # shows dots instead of the key
            help="Leave empty if the key is already set on the server "
                 "(GEMINI_API_KEY environment variable or Streamlit secret).",
        )
        top_k = st.slider("Chunks used per answer", 1, 8, DEFAULT_TOP_K)
        min_score = st.slider("Minimum match score", 0.0, 1.0, DEFAULT_MIN_SCORE, 0.05,
                              help="If the best chunk scores below this, the app says it doesn't know.")
        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()                       # redraw the page right away

    # ---- Need an API key before anything else ------------------------------
    client = get_client(typed_key)
    if client is None:
        st.info("Enter your Gemini API key in the sidebar to start.")
        st.stop()                            # stop here; nothing below runs

    # ---- File upload --------------------------------------------------------
    uploaded = st.file_uploader("Upload a document", type=["pdf", "txt", "md"])
    if uploaded is None:
        st.info("Upload a PDF, .txt or .md file to begin.")
        st.stop()

    data = uploaded.getvalue()                       # the file as raw bytes
    file_id = hashlib.sha256(data).hexdigest()       # fingerprint: same file = same id

    # ---- Process the file ONLY when it is new ------------------------------
    # Without this check, the (slow, API-costing) embedding step would repeat
    # every time you typed a question, because the script re-runs each time.
    if st.session_state.get("file_id") != file_id:
        try:
            pages = extract_pages(data)
        except Exception as error:
            st.error(f"Could not read this file: {error}")
            st.stop()

        if not pages:
            st.error("No readable text found. If this is a scanned PDF (only images), "
                     "it needs OCR first.")
            st.stop()

        chunks = build_chunks(pages, is_pdf_bytes(data))
        if len(chunks) > MAX_CHUNKS:
            st.error(f"This file is too large ({len(chunks)} chunks, limit {MAX_CHUNKS}). "
                     "Please upload a smaller file.")
            st.stop()

        try:
            vectors = build_index(client, chunks)    # slow step: one API call per chunk
        except Exception as error:
            st.error(f"Could not embed the document: {error}")
            st.stop()

        # Save everything so the next re-run can reuse it, and reset the chat
        st.session_state.update(
            file_id=file_id, file_name=uploaded.name,
            chunks=chunks, vectors=vectors, messages=[],
        )

    chunks = st.session_state.chunks
    vectors = st.session_state.vectors
    st.success(f"Ready: {st.session_state.file_name} ({len(chunks)} chunks). Ask your question below.")

    # ---- Redraw the earlier messages (history) ------------------------------
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):       # role is "user" or "assistant"
            st.markdown(message["content"])
            show_sources(message.get("sources"))

    # ---- The chat box: this is where you type your question ----------------
    question = st.chat_input("Ask a question about the document")
    if question:                                     # runs only after you press Enter
        # Save and show the user's question
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Find the answer and show it
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                sources = []
                try:
                    hits = retrieve(client, question, chunks, vectors, top_k)
                    if hits[0][1] < min_score:       # best match is too weak
                        reply = "I don't know based on the document."
                    else:
                        reply = answer(client, question, hits)
                        sources = [{"label": c["label"], "score": s, "text": c["text"]}
                                   for c, s in hits]
                except Exception as error:           # show errors in the chat instead of crashing
                    reply = f"Something went wrong: {error}"
            st.markdown(reply)
            show_sources(sources)

        # Save the answer so it is redrawn on the next re-run
        st.session_state.messages.append(
            {"role": "assistant", "content": reply, "sources": sources}
        )


# Streamlit runs this file as the main program, so this starts the app
if __name__ == "__main__":
    main()