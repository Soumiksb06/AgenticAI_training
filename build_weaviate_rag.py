import os
from pathlib import Path
import weaviate
from weaviate.classes.config import DataType, Property
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# LangChain Document Loaders
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredFileLoader
)

DOCUMENTS_DIR = "rag/documents"

def extract_text_from_file(file_path: str) -> str:
    """Dynamically routes the file to the correct LangChain loader based on extension."""
    ext = Path(file_path).suffix.lower()
    
    try:
        if ext == '.pdf':
            loader = PyPDFLoader(file_path)
        elif ext in ['.txt', '.md', '.csv', '.json']:
            loader = TextLoader(file_path, autodetect_encoding=True)
        else:
            # Fallback for .docx, .pptx, .html, and other file types
            loader = UnstructuredFileLoader(file_path)
            
        docs = loader.load()
        return "\n".join([doc.page_content for doc in docs])
    except Exception as e:
        print(f"      [!] Error reading file: {e}")
        return ""

def build_weaviate_rag():
    print("=" * 75)
    print("BUILDING WEAVIATE RAG KNOWLEDGE BASE (UNIVERSAL FILE SUPPORT)")
    print("=" * 75)

    # 1. Load local embedding model in Python
    print("\n1. Loading local SentenceTransformer model ('all-MiniLM-L6-v2')...")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # 2. Connect to local Weaviate instance
    client = weaviate.connect_to_local()

    try:
        if client.collections.exists("InsuranceKnowledge"):
            client.collections.delete("InsuranceKnowledge")

        # Create collection (defaults to 'none' vectorizer, avoiding deprecation warnings)
        collection = client.collections.create(
            name="InsuranceKnowledge",
            properties=[
                Property(name="content", data_type=DataType.TEXT),
                Property(name="category", data_type=DataType.TEXT),
                Property(name="source_file", data_type=DataType.TEXT),
            ]
        )
        print("[✓] Created 'InsuranceKnowledge' collection in Weaviate.")

        # 3. Locate ALL files across subfolders (ignoring hidden system files)
        os.makedirs(DOCUMENTS_DIR, exist_ok=True)
        all_files = []
        for root, _, files in os.walk(DOCUMENTS_DIR):
            for file in files:
                if not file.startswith('.'):  # Ignore .DS_Store, etc.
                    all_files.append(os.path.join(root, file))

        if not all_files:
            print(f"\n[!] Notice: No documents found under '{DOCUMENTS_DIR}'.")
            print("    Skipping document ingestion. Empty 'InsuranceKnowledge' collection is ready.")
            return

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=60,
            separators=["SECTION:", "CASE ID:", "\n\n", "\n", "- ", " "]
        )

        kb_collection = client.collections.use("InsuranceKnowledge")

        print(f"\n2. Found {len(all_files)} document file(s). Parsing, chunking, and ingesting:")
        with kb_collection.batch.dynamic() as batch:
            for file_path in all_files:
                category = Path(file_path).parent.name
                file_name = Path(file_path).name

                # Extract text dynamically based on file type
                raw_text = extract_text_from_file(file_path)

                if not raw_text.strip():
                    print(f"   [-] Category: '{category}' | File: '{file_name}' | Empty or unreadable file, skipping.")
                    continue

                chunks = text_splitter.split_text(raw_text)
                print(f"   [+] Category: '{category}' | File: '{file_name}' | Chunks: {len(chunks)}")

                for chunk in chunks:
                    vector = embedder.encode(chunk).tolist()

                    batch.add_object(
                        properties={
                            "content": chunk,
                            "category": category,
                            "source_file": file_name
                        },
                        vector=vector
                    )

        print("\n[✓] Successfully ingested all available document chunks into Weaviate!")

    finally:
        client.close()

if __name__ == "__main__":
    build_weaviate_rag()