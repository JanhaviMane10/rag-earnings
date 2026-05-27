# ============================================================
# 01_build_rag.py — Build the RAG Pipeline
#
# WHAT THIS FILE DOES:
# 1. Takes earnings call transcripts
# 2. Splits them into small chunks
# 3. Converts chunks to embeddings (numbers that capture meaning)
# 4. Stores everything in ChromaDB (vector database)
# 5. Also stores metadata in SQLite (company, quarter, date)
#
# WHAT IS AN EMBEDDING?
# Think of it like GPS coordinates but for meaning.
# "Apple AI revenue" and "iPhone AI monetization" will have
# similar coordinates even though the words are different.
# This lets us find relevant text by meaning, not just keywords.
# ============================================================

# %% — Load libraries
import os
import json
import sqlite3
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import warnings
warnings.filterwarnings("ignore")

from config import *

print("✅ Libraries loaded")

# %% — Load transcripts
print("\n📄 Loading earnings call transcripts...")
transcripts = SAMPLE_TRANSCRIPTS

print(f"✅ Loaded {len(transcripts)} transcripts:")
for key in transcripts:
    print(f"   - {key}")

# %% — Create SQLite metadata database
print("\n🗄️  Creating SQLite metadata database...")
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS transcripts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id      TEXT UNIQUE,
    company     TEXT,
    ticker      TEXT,
    quarter     TEXT,
    year        INTEGER,
    revenue_bn  REAL,
    yoy_growth  REAL,
    key_topics  TEXT,
    n_chunks    INTEGER
)
""")

# Metadata for each transcript
metadata_map = {
    "AAPL_Q4_2024": {"company": "Apple",     "ticker": "AAPL", "quarter": "Q4", "year": 2024, "revenue_bn": 94.9, "yoy_growth": 6.0,   "key_topics": "iPhone,Services,Apple Intelligence,AI"},
    "MSFT_Q1_2025": {"company": "Microsoft", "ticker": "MSFT", "quarter": "Q1", "year": 2025, "revenue_bn": 65.6, "yoy_growth": 16.0,  "key_topics": "Azure,AI,Copilot,Cloud"},
    "GOOGL_Q3_2024":{"company": "Google",    "ticker": "GOOGL","quarter": "Q3", "year": 2024, "revenue_bn": 88.3, "yoy_growth": 15.0,  "key_topics": "Search,YouTube,Cloud,Gemini,AI"},
    "NVDA_Q2_2025": {"company": "NVIDIA",    "ticker": "NVDA", "quarter": "Q2", "year": 2025, "revenue_bn": 30.0, "yoy_growth": 122.0, "key_topics": "Data Center,Blackwell,AI,GPU"},
    "META_Q3_2024": {"company": "Meta",      "ticker": "META", "quarter": "Q3", "year": 2024, "revenue_bn": 40.6, "yoy_growth": 19.0,  "key_topics": "Advertising,AI,Llama,VR,Social Media"},
}

conn.commit()
print("✅ SQLite database created")

# %% — Split transcripts into chunks
print("\n✂️  Splitting transcripts into chunks...")
print(f"   Chunk size: {CHUNK_SIZE} characters")
print(f"   Chunk overlap: {CHUNK_OVERLAP} characters")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ".", " "]
)

all_chunks  = []
all_metadata = []

for doc_id, text in transcripts.items():
    meta = metadata_map.get(doc_id, {})
    chunks = splitter.split_text(text.strip())

    for i, chunk in enumerate(chunks):
        all_chunks.append(chunk)
        all_metadata.append({
            "doc_id":   doc_id,
            "company":  meta.get("company", "Unknown"),
            "ticker":   meta.get("ticker",  "N/A"),
            "quarter":  meta.get("quarter", "N/A"),
            "year":     str(meta.get("year", 0)),
            "chunk_id": f"{doc_id}_chunk_{i}",
            "topics":   meta.get("key_topics", ""),
        })

    # Store in SQLite
    cursor.execute("""
    INSERT OR REPLACE INTO transcripts
    (doc_id, company, ticker, quarter, year, revenue_bn, yoy_growth, key_topics, n_chunks)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (doc_id, meta.get("company"), meta.get("ticker"),
          meta.get("quarter"), meta.get("year"),
          meta.get("revenue_bn"), meta.get("yoy_growth"),
          meta.get("key_topics"), len(chunks)))

    print(f"   ✅ {doc_id}: {len(chunks)} chunks")

conn.commit()
conn.close()

print(f"\n✅ Total chunks created: {len(all_chunks)}")

# %% — Create embeddings and store in ChromaDB
print("\n🧠 Creating embeddings and building vector database...")
print("   (This converts text chunks into numerical vectors)")
print("   Using model: all-MiniLM-L6-v2 (free, runs locally)")
print("   Please wait — first run downloads the model (~80MB)...\n")

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# Create ChromaDB vector store
vectorstore = Chroma.from_texts(
    texts=all_chunks,
    embedding=embedding_model,
    metadatas=all_metadata,
    persist_directory=CHROMA_DIR,
    collection_name="earnings_calls"
)

print(f"✅ Vector database created at {CHROMA_DIR}")
print(f"   {len(all_chunks)} chunks stored with embeddings")

# %% — Test the retrieval
print("\n🔍 Testing retrieval with a sample query...")
test_query = "What did management say about AI revenue and growth?"

results = vectorstore.similarity_search(test_query, k=3)

print(f"\nQuery: '{test_query}'")
print(f"\nTop 3 most relevant chunks:")
for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(f"Source: {doc.metadata.get('company')} {doc.metadata.get('quarter')} {doc.metadata.get('year')}")
    print(f"Text: {doc.page_content[:200]}...")

# %% — Save chunk info for dashboard
chunks_df = pd.DataFrame([
    {"doc_id": m["doc_id"], "company": m["company"],
     "ticker": m["ticker"], "quarter": m["quarter"],
     "year": m["year"], "chunk_text": c[:100] + "..."}
    for c, m in zip(all_chunks, all_metadata)
])
chunks_df.to_csv(f"{OUTPUT_DIR}/chunks_index.csv", index=False)

print(f"\n✅ Chunk index saved to {OUTPUT_DIR}/chunks_index.csv")
print(f"\n" + "="*60)
print("📋 RAG BUILD SUMMARY")
print("="*60)
print(f"✅ Transcripts processed: {len(transcripts)}")
print(f"✅ Total chunks: {len(all_chunks)}")
print(f"✅ Vector DB: {CHROMA_DIR}")
print(f"✅ Metadata DB: {DB_PATH}")
print(f"\n👉 Next step: Run 02_evaluate_rag.py to test quality")
