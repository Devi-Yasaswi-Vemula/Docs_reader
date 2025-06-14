import torch

# Prevent streamlit watcher from breaking on torch.classes
import types
torch.classes = types.SimpleNamespace(__path__=[])

import os
import uuid
import shutil
import sqlite3
from datetime import datetime
import easyocr

import pytesseract
import numpy as np
import easyocr
from PIL import Image
from pdf2image import convert_from_path

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

import streamlit as st
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredHTMLLoader
from langchain.docstore.document import Document
from pdf2image import convert_from_path
from PIL import Image
import easyocr
reader = easyocr.Reader(['en'])


# Set Tesseract executable path
# pytesseract.pytesseract.tesseract_cmd = r'C:\Users\e1775060\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

# -----------------------------
# 1. ENV & CONFIG
# -----------------------------
load_dotenv()
DB_NAME = "rag_app.db"
CHROMA_DIR = "./chroma_db"
os.makedirs(CHROMA_DIR, exist_ok=True)

# -----------------------------
# 2. CHROMA VECTOR STORE
# -----------------------------
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
embedding_function = OpenAIEmbeddings()
vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding_function)

# def extract_text_with_easyocr(image):
#     result = reader.readtext(image, detail=0)
#     return " ".join(result)

def extract_text_with_easyocr(image):
    reader = easyocr.Reader(['en'])
    result = reader.readtext(np.array(image), detail=0)
    return "\n".join(result)

def load_and_split_document(file_path):
    documents = []

    if file_path.endswith('.pdf'):
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()

            # Check if text is actually present
            if not any(doc.page_content.strip() for doc in documents):
                raise ValueError("Empty PDF content")

        except Exception:
            st.warning("PDF appears image-based. Using OCR...")
            try:
                images = convert_from_path(
                    file_path,
                    poppler_path=r'C:\Users\e1775060\Downloads\Release-24.08.0-0\poppler-24.08.0\Library\bin'
                )
                for i, img in enumerate(images):
                    text = extract_text_with_easyocr(img)
                    if text.strip():
                        documents.append(
                            Document(
                                page_content=text,
                                metadata={"source": f"{file_path} - Page {i+1}"}
                            )
                        )
                    else:
                        st.warning(f"No text found on page {i+1} using OCR.")
            except Exception as e:
                st.error(f"OCR failed on PDF pages: {e}")
                return []

    elif file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        try:
            image = Image.open(file_path)
            text = extract_text_with_easyocr(image)
            if text.strip():
                documents = [Document(page_content=text, metadata={"source": file_path})]
            else:
                st.warning("No text found in the image using OCR.")
        except Exception as e:
            st.error(f"Image OCR failed: {e}")
            return []

    elif file_path.endswith('.docx'):
        try:
            loader = Docx2txtLoader(file_path)
            documents = loader.load()
        except Exception as e:
            st.error(f"DOCX loading failed: {e}")
            return []

    elif file_path.endswith('.html'):
        try:
            loader = UnstructuredHTMLLoader(file_path)
            documents = loader.load()
        except Exception as e:
            st.error(f"HTML loading failed: {e}")
            return []

    else:
        raise ValueError("Unsupported file type.")

    # Return the split documents using your text splitter
    return text_splitter.split_documents(documents)
# def load_and_split_document(file_path):
#     if file_path.endswith('.pdf'):
#         try:
#             loader = PyPDFLoader(file_path)
#             documents = loader.load()
#             if not any(doc.page_content.strip() for doc in documents):
#                 raise ValueError("Empty content")
#         except Exception:
#             st.warning("PDF appears image-based. Trying OCR...")
#             images = convert_from_path(file_path, poppler_path=r'C:\Users\e1775060\Downloads\Release-24.08.0-0\poppler-24.08.0\Library\bin')
#             documents = []
#             for i, img in enumerate(images):
#                 try:
#                     # text = pytesseract.image_to_string(img)
#                     reader = easyocr.Reader(['en'])
#                     result = reader.readtext(image_path_or_np_array, detail=0)
#                     text = " ".join(result)
#                     if not text.strip():
#                         text = extract_text_with_easyocr(img)
#                     documents.append(Document(page_content=text, metadata={"source": f"{file_path} - Page {i+1}"}))
#                 except Exception as e:
#                     st.error(f"OCR failed on page {i+1}: {e}")
#     elif file_path.endswith(('.png', '.jpg', '.jpeg')):
#         try:
#             image = Image.open(file_path)
#             text = pytesseract.image_to_string(image)
#             if not text.strip():
#                 text = extract_text_with_easyocr(image)
#             documents = [Document(page_content=text, metadata={"source": file_path})]
#         except Exception as e:
#             st.error(f"Image OCR failed: {e}")
#             return []
#     elif file_path.endswith('.docx'):
#         loader = Docx2txtLoader(file_path)
#         documents = loader.load()
#     elif file_path.endswith('.html'):
#         loader = UnstructuredHTMLLoader(file_path)
#         documents = loader.load()
#     else:
#         raise ValueError("Unsupported file type.")

#     return text_splitter.split_documents(documents)


def index_document(file_path, file_id):
    try:
        splits = load_and_split_document(file_path)
        for split in splits:
            split.metadata["file_id"] = file_id
        vectorstore.add_documents(splits)
        return True
    except Exception as e:
        st.error(f"Error indexing document: {str(e)}")
        return False


def delete_vector_docs(file_id):
    try:
        vectorstore._collection.delete(where={"file_id": file_id})
        return True
    except Exception as e:
        st.error(f"Error deleting from Chroma: {str(e)}")
        return False

# -----------------------------
# 3. SQLITE UTILS
# -----------------------------
def get_conn():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    conn.execute('''CREATE TABLE IF NOT EXISTS document_store (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.execute('''CREATE TABLE IF NOT EXISTS application_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        user_query TEXT,
        gpt_response TEXT,
        model TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.close()

def insert_document_record(filename):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("INSERT INTO document_store (filename) VALUES (?)", (filename,))
    file_id = cur.lastrowid
    conn.commit()
    conn.close()
    return file_id

def delete_document_record(file_id):
    conn = get_conn()
    conn.execute("DELETE FROM document_store WHERE id = ?", (file_id,))
    conn.commit()
    conn.close()

def get_all_documents():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM document_store ORDER BY upload_timestamp DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]

def insert_log(session_id, question, answer, model):
    conn = get_conn()
    conn.execute("INSERT INTO application_logs (session_id, user_query, gpt_response, model) VALUES (?, ?, ?, ?)",
                 (session_id, question, answer, model))
    conn.commit()
    conn.close()

def get_chat_history(session_id):
    conn = get_conn()
    rows = conn.execute("SELECT user_query, gpt_response FROM application_logs WHERE session_id = ? ORDER BY created_at", (session_id,))
    history = []
    for r in rows:
        history.extend([
            {"role": "human", "content": r["user_query"]},
            {"role": "ai", "content": r["gpt_response"]},
        ])
    conn.close()
    return history

def get_rag_chain(model_name="gpt-4o-mini", file_id=None):
    llm = ChatOpenAI(model=model_name)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3, "filter": {"file_id": file_id}}) if file_id else vectorstore.as_retriever(search_kwargs={"k": 3})
    output_parser = StrOutputParser()
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given chat history and a new question, return a standalone question."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use the following context to answer."),
        ("system", "Context: {context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, contextualize_prompt)
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(retriever_chain, qa_chain)

# -----------------------------
# 4. STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="PDF Chat App", layout="wide")
# st.title("\ud83d\udcc4 Chat with Your Documents")
st.title("📄 Chat with Your Documents")

init_db()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "model" not in st.session_state:
    st.session_state.model = "gpt-4o-mini"

# Sidebar: Upload & List
st.sidebar.header("📄 Document Manager")
with st.sidebar.form("upload_form", clear_on_submit=True):
    file = st.file_uploader("Upload PDF/DOCX/HTML/Image", type=["pdf", "docx", "html", "png", "jpg", "jpeg"])
    if st.form_submit_button("Upload & Index") and file:
        temp_path = f"temp_{file.name}"
        with open(temp_path, "wb") as f:
            f.write(file.read())
        file_id = insert_document_record(file.name)
        success = index_document(temp_path, file_id)
        os.remove(temp_path)
        if success:
            st.sidebar.success(f"Uploaded and indexed {file.name}")
        else:
            delete_document_record(file_id)
            st.sidebar.error("Failed to index file.")

st.sidebar.subheader("📄 Uploaded Files")
for doc in get_all_documents():
    col1, col2 = st.sidebar.columns([4, 1])
    with col1:
        st.markdown(f"📄 {doc['filename']}")
    with col2:
        if st.button("📄", key=f"del_{doc['id']}"):
            delete_vector_docs(doc['id'])
            delete_document_record(doc['id'])
            st.rerun()

# Chat Interface
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Thinking..."):
        file_records = get_all_documents()
        latest_file_id = file_records[0]['id'] if file_records else None
        chain = get_rag_chain(st.session_state.model, file_id=latest_file_id)
        history = get_chat_history(st.session_state.session_id)
        result = chain.invoke({"input": prompt, "chat_history": history})
        answer = result.get("answer", "No relevant answer found.")
        st.session_state.messages.append({"role": "assistant", "content": answer})
        insert_log(st.session_state.session_id, prompt, answer, st.session_state.model)
        with st.chat_message("assistant"):
            st.markdown(answer)
        with st.expander("Response Details"):
            st.code(answer, language="markdown")
            st.code(f"Session ID: {st.session_state.session_id}")
            st.code(f"Model: {st.session_state.model}")