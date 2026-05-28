# 📈 Earnings Call Analyst — RAG System
### Chat with Financial Earnings Calls using Llama 3.3 70B + ChromaDB

[![Python](https://img.shields.io/badge/Python-3.x-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20Demo-red)](https://janhavi-rag-earnings.streamlit.app)
[![LLM](https://img.shields.io/badge/LLM-Llama%203.3%2070B-purple)]()

🔗 **[Live Demo](https://janhavi-rag-earnings.streamlit.app)**

---

## 📋 Project Overview

Ask any question about earnings calls from Apple, Microsoft, Google, NVIDIA, and Meta — and get grounded, cited answers with confidence scores. This project implements a full **Retrieval Augmented Generation (RAG)** pipeline with hallucination detection.

> **Business Question:** What did NVIDIA's CEO say about Blackwell chip supply constraints? How does Microsoft's AI monetization compare to Google's?

---

## 🔍 Key Features

- **Chat with memory** — conversation history carried across questions
- **Compare mode** — ask the same question about two companies side by side
- **Confidence scores** — every answer rated 0-100% with color-coded confidence bar
- **Hallucination detection** — flags answers not grounded in source documents
- **SQL metadata explorer** — query document metadata directly

---

## 📊 System Performance

| Metric | Value |
|---|---|
| Grounding Rate | 87.5% |
| Hallucination Rate | 12.5% |
| Source Retrieval Accuracy | 100% |
| Model | Llama 3.3 70B via Groq |
| Embedding Model | all-MiniLM-L6-v2 |
| Vector DB | ChromaDB |
| Documents | 5 company earnings calls |

---

## 🗂️ Repository Structure

```
rag-earnings/
├── config.py              ← Settings + sample transcripts
├── 01_build_rag.py        ← Build vector store + SQLite metadata DB
├── 02_evaluate_rag.py     ← Evaluate RAG quality (8 test questions)
├── app.py                 ← Streamlit chat dashboard
├── requirements.txt
└── outputs/
    ├── eval_results.csv
    ├── eval_summary.json
    └── chunks_index.csv
```

---

## 🚀 How to Run

```bash
git clone https://github.com/JanhaviMane10/rag-earnings.git
cd rag-earnings

pip install -r requirements.txt

# Add your Groq API key (free at console.groq.com)
echo "GROQ_API_KEY=your_key_here" > .env

python 01_build_rag.py
python 02_evaluate_rag.py
streamlit run app.py
```

---

## 🛠️ Methods

**RAG Pipeline:**
1. **Chunk** — Split transcripts into 500-char chunks with 50-char overlap
2. **Embed** — Convert chunks to vectors using `all-MiniLM-L6-v2`
3. **Store** — Save embeddings in ChromaDB, metadata in SQLite
4. **Retrieve** — Find top-5 most semantically similar chunks
5. **Generate** — Llama 3.3 70B answers using ONLY retrieved context
6. **Verify** — Second LLM call checks if answer is grounded in context

**Evaluation:**
- 8 test questions with known answers
- Measures source retrieval accuracy, answer accuracy, grounding rate
- Includes out-of-scope questions to test refusal behavior

---

## 💡 Companies Covered

| Company | Quarter | Revenue | YoY Growth |
|---|---|---|---|
| Apple | Q4 2024 | $94.9B | +6% |
| Microsoft | Q1 2025 | $65.6B | +16% |
| Google | Q3 2024 | $88.3B | +15% |
| NVIDIA | Q2 2025 | $30.0B | +122% |
| Meta | Q3 2024 | $40.6B | +19% |

---

## 🔧 Technologies

`Python` `LangChain` `ChromaDB` `Groq` `Llama 3.3 70B` `HuggingFace` `SQLite` `Streamlit` `Plotly`
