# modules/rag_pipeline.py
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.document_loaders import PyPDFLoader

# -----------------------------
# Document Processing + FAISS DB
# -----------------------------

def build_vector_db(pdf_path: str, db_path: str = "vectorstore") -> FAISS:
    """
    Loads a PDF, chunks it, embeds, and stores in FAISS vector database.
    If FAISS DB already exists, it loads from disk.
    """
    if os.path.exists(db_path):
        return FAISS.load_local(db_path, SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"), allow_dangerous_deserialization=True)

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # Chunk text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    # Embeddings
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Store in FAISS
    db = FAISS.from_documents(splits, embeddings)
    db.save_local(db_path)

    return db


def retrieve_docs(query: str, db: FAISS, k: int = 3):
    """
    Retrieves top-k relevant documents for a query.
    """
    return db.similarity_search(query, k=k)
