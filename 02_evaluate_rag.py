# ============================================================
# 02_evaluate_rag.py — Evaluate RAG Quality
#
# WHAT THIS FILE DOES:
# Tests the RAG system on a set of questions and measures:
# 1. Retrieval accuracy — did we find the right chunks?
# 2. Answer relevance — is the answer actually useful?
# 3. Hallucination detection — did the model make things up?
#
# WHY THIS MATTERS FOR YOUR PORTFOLIO:
# Most RAG projects stop at "it answers questions."
# Adding evaluation is what separates you from 90% of candidates.
# Real companies need to know: "Can we trust this system?"
# ============================================================

# %% — Load libraries
import os
import json
import sqlite3
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
import warnings
warnings.filterwarnings("ignore")

from config import *

print("✅ Libraries loaded")

# %% — Load vector store and LLM
print("\n🔌 Loading vector store and LLM...")

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embedding_model,
    collection_name="earnings_calls"
)

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name=LLM_MODEL,
    temperature=0.1,   # Low temperature = more factual
    max_tokens=500
)

print(f"✅ Vector store loaded: {vectorstore._collection.count()} chunks")
print(f"✅ LLM loaded: {LLM_MODEL}")

# %% — Define evaluation questions with known answers
print("\n📝 Setting up evaluation questions...")

eval_questions = [
    {
        "question":        "What was Apple's revenue in Q4 2024?",
        "expected_source": "AAPL_Q4_2024",
        "expected_answer": "94.9 billion",
        "category":        "Financial Facts"
    },
    {
        "question":        "How many paid subscriptions does Apple have?",
        "expected_source": "AAPL_Q4_2024",
        "expected_answer": "over 1 billion",
        "category":        "Business Metrics"
    },
    {
        "question":        "What was Microsoft Azure's growth rate?",
        "expected_source": "MSFT_Q1_2025",
        "expected_answer": "33%",
        "category":        "Financial Facts"
    },
    {
        "question":        "How many GitHub Copilot paid subscribers does Microsoft have?",
        "expected_source": "MSFT_Q1_2025",
        "expected_answer": "1.8 million",
        "category":        "Business Metrics"
    },
    {
        "question":        "What was NVIDIA's revenue growth year over year?",
        "expected_source": "NVDA_Q2_2025",
        "expected_answer": "122%",
        "category":        "Financial Facts"
    },
    {
        "question":        "How many monthly active users does Meta AI have?",
        "expected_source": "META_Q3_2024",
        "expected_answer": "500 million",
        "category":        "Business Metrics"
    },
    {
        "question":        "What did Google say about AI Overviews user numbers?",
        "expected_source": "GOOGL_Q3_2024",
        "expected_answer": "1 billion users",
        "category":        "AI Strategy"
    },
    {
        "question":        "What is the weather in New York today?",
        "expected_source": None,
        "expected_answer": "NOT IN DOCUMENTS",
        "category":        "Out of Scope (should decline)"
    },
]

print(f"✅ {len(eval_questions)} evaluation questions ready")

# %% — RAG pipeline function
def ask_rag(question, k=TOP_K):
    """
    Full RAG pipeline:
    1. Retrieve relevant chunks
    2. Build prompt with context
    3. Generate answer
    4. Check if answer is grounded in context
    """
    # Step 1: Retrieve
    docs = vectorstore.similarity_search(question, k=k)
    context = "\n\n---\n\n".join([
        f"[{d.metadata.get('company')} {d.metadata.get('quarter')} {d.metadata.get('year')}]\n{d.page_content}"
        for d in docs
    ])
    sources = list(set([d.metadata.get("doc_id") for d in docs]))

    # Step 2: Generate answer
    system_prompt = """You are a financial analyst assistant. Answer questions based ONLY on the provided earnings call transcripts.

Rules:
1. Only use information from the provided context
2. If the answer is not in the context, say "This information is not available in the provided transcripts"
3. Always cite which company and quarter your answer comes from
4. Be concise and factual
5. Never make up numbers or facts"""

    user_prompt = f"""Context from earnings call transcripts:
{context}

Question: {question}

Answer based only on the context above:"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    response = llm.invoke(messages)
    answer = response.content

    # Step 3: Hallucination check
    hallucination_prompt = f"""Given this context:
{context[:1000]}

And this answer:
{answer}

Is the answer fully supported by the context? Reply with only YES or NO."""

    hal_messages = [HumanMessage(content=hallucination_prompt)]
    hal_response = llm.invoke(hal_messages)
    is_grounded  = "YES" in hal_response.content.upper()

    return {
        "answer":       answer,
        "sources":      sources,
        "context":      context,
        "is_grounded":  is_grounded,
        "n_chunks":     len(docs),
    }

# %% — Run evaluation
print("\n🧪 Running evaluation...")
print("="*60)

results = []

for i, q in enumerate(eval_questions):
    print(f"\n[{i+1}/{len(eval_questions)}] {q['question']}")

    result = ask_rag(q["question"])

    # Check if correct source was retrieved
    source_found = (q["expected_source"] is None or
                    any(q["expected_source"] in s for s in result["sources"]))

    # Check if expected answer appears in response
    answer_correct = (q["expected_answer"].lower() in result["answer"].lower() or
                      q["expected_answer"] == "NOT IN DOCUMENTS")

    results.append({
        "question":       q["question"],
        "category":       q["category"],
        "expected_source":q["expected_source"],
        "retrieved_sources": ", ".join(result["sources"]),
        "source_correct": source_found,
        "answer":         result["answer"][:200],
        "expected_answer":q["expected_answer"],
        "answer_correct": answer_correct,
        "is_grounded":    result["is_grounded"],
    })

    status = "✅" if (source_found and result["is_grounded"]) else "⚠️"
    print(f"   {status} Source correct: {source_found} | Grounded: {result['is_grounded']}")
    print(f"   Answer: {result['answer'][:150]}...")

# %% — Evaluation metrics
print("\n" + "="*60)
print("📊 EVALUATION RESULTS")
print("="*60)

results_df = pd.DataFrame(results)

source_accuracy   = results_df["source_correct"].mean() * 100
answer_accuracy   = results_df["answer_correct"].mean() * 100
grounding_rate    = results_df["is_grounded"].mean() * 100
hallucination_rate= 100 - grounding_rate

print(f"\n✅ Source Retrieval Accuracy:  {source_accuracy:.1f}%")
print(f"✅ Answer Accuracy:            {answer_accuracy:.1f}%")
print(f"✅ Grounding Rate:             {grounding_rate:.1f}%")
print(f"⚠️  Hallucination Rate:         {hallucination_rate:.1f}%")

print(f"\nResults by category:")
cat_results = results_df.groupby("category").agg(
    n=("question", "count"),
    source_acc=("source_correct", "mean"),
    grounding=("is_grounded", "mean")
).reset_index()
cat_results["source_acc"] = (cat_results["source_acc"] * 100).round(1)
cat_results["grounding"]  = (cat_results["grounding"]  * 100).round(1)
print(cat_results.to_string(index=False))

# Save results
results_df.to_csv(f"{OUTPUT_DIR}/eval_results.csv", index=False)

eval_summary = {
    "source_accuracy":    float(source_accuracy),
    "answer_accuracy":    float(answer_accuracy),
    "grounding_rate":     float(grounding_rate),
    "hallucination_rate": float(hallucination_rate),
    "n_questions":        len(eval_questions),
    "model":              LLM_MODEL,
    "embedding_model":    EMBEDDING_MODEL,
    "top_k":              TOP_K,
    "chunk_size":         CHUNK_SIZE,
}
with open(f"{OUTPUT_DIR}/eval_summary.json", "w") as f:
    json.dump(eval_summary, f, indent=2)

print(f"\n✅ Results saved to {OUTPUT_DIR}/eval_results.csv")
print(f"\n👉 Next step: Run streamlit run app.py to launch the dashboard")
