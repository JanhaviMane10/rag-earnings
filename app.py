# ============================================================
# app.py — Earnings Call RAG Dashboard (v2)
# Upgraded: Chat history, Compare mode, Confidence scores
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sqlite3
import json
import os
import time
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from config import *

st.set_page_config(
    page_title="Earnings Call Analyst",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
    .stApp { background-color: #0F1117; }
    [data-testid="stSidebar"] { background-color: #1A1D27; border-right: 1px solid #2D2F3E; }
    .metric-card { background: linear-gradient(135deg, #1E2235 0%, #252840 100%); border: 1px solid #3D4070; border-radius: 12px; padding: 20px; text-align: center; }
    .metric-value { font-size: 28px; font-weight: 700; background: linear-gradient(90deg, #6C63FF, #48CAE4); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .metric-label { font-size: 11px; color: #8B8FA8; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.08em; }
    .section-header { font-size: 11px; font-weight: 600; color: #6C63FF; text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 12px; padding-bottom: 8px; border-bottom: 1px solid #2D2F3E; }
    .hero-banner { background: linear-gradient(135deg, #1A1D2E 0%, #252840 50%, #1E2235 100%); border: 1px solid #3D4070; border-radius: 16px; padding: 28px 32px; margin-bottom: 24px; }
    .hero-title { font-size: 26px; font-weight: 700; color: #E0E3F0; margin: 0; }
    .hero-subtitle { font-size: 14px; color: #8B8FA8; margin-top: 6px; }
    .badge { font-size: 11px; padding: 4px 12px; border-radius: 20px; font-weight: 500; display: inline-block; margin: 4px; }
    .badge-purple { background: #2D2B55; color: #A5A0FF; border: 1px solid #4D4A8A; }
    .badge-blue   { background: #1A2D44; color: #7EC8E3; border: 1px solid #2A4D6A; }
    .badge-green  { background: #1A2E1E; color: #86EFAC; border: 1px solid #2A4E2E; }
    .badge-orange { background: #2E1E0A; color: #FDC98A; border: 1px solid #4E3010; }

    /* Chat bubbles */
    .user-bubble { background: #2D2B55; border-radius: 12px 12px 2px 12px; padding: 12px 16px; margin: 8px 0; margin-left: 20%; color: #E0E3F0; font-size: 14px; }
    .bot-bubble { background: linear-gradient(135deg, #1A2744 0%, #1E2D4A 100%); border: 1px solid #3D4070; border-radius: 12px 12px 12px 2px; padding: 14px 18px; margin: 8px 0; margin-right: 20%; }
    .bot-answer { font-size: 14px; color: #E0E3F0; line-height: 1.7; }
    .bot-meta { margin-top: 10px; display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
    .source-tag { display: inline-block; background: #2D2B55; color: #A5A0FF; padding: 2px 10px; border-radius: 12px; font-size: 11px; }
    .confidence-bar { height: 4px; border-radius: 2px; margin-top: 6px; }

    /* Compare mode */
    .compare-col { background: linear-gradient(135deg, #1A1D2E 0%, #1E2235 100%); border: 1px solid #2D2F3E; border-radius: 12px; padding: 16px; }
    .compare-header { font-size: 13px; font-weight: 600; color: #A5A0FF; margin-bottom: 10px; }

    .chunk-card { background: #1A1D27; border: 1px solid #2D2F3E; border-radius: 8px; padding: 12px 16px; margin: 6px 0; font-size: 13px; color: #C8CCE0; }
    .highlight { background: rgba(108,99,255,0.25); padding: 2px 4px; border-radius: 3px; }
    .stTabs [data-baseweb="tab-list"] { background: #1A1D27; border-radius: 10px; padding: 4px; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px; color: #8B8FA8; font-size: 13px; }
    .stTabs [aria-selected="true"] { background: #252840 !important; color: #E0E3F0 !important; }
    hr { border-color: #2D2F3E; }
</style>
""", unsafe_allow_html=True)

PLOT_TEMPLATE = "plotly_dark"
COLORS = {"primary": "#6C63FF", "secondary": "#48CAE4", "success": "#4ADE80",
          "warning": "#FBBF24", "danger": "#F87171", "neutral": "#4A4E6A"}

# ── Session state ─────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "compare_q" not in st.session_state:
    st.session_state.compare_q = ""

# ── Load models ───────────────────────────────────────────────
@st.cache_resource
def load_rag():
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    # Build vector store from scratch if it doesn't exist
    if not os.path.exists(CHROMA_DIR) or len(os.listdir(CHROMA_DIR)) == 0:
        os.makedirs(CHROMA_DIR, exist_ok=True)
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        all_chunks, all_metadata = [], []
        meta_map = {
            "AAPL_Q4_2024":  {"company":"Apple",     "ticker":"AAPL",  "quarter":"Q4","year":"2024"},
            "MSFT_Q1_2025":  {"company":"Microsoft", "ticker":"MSFT",  "quarter":"Q1","year":"2025"},
            "GOOGL_Q3_2024": {"company":"Google",    "ticker":"GOOGL", "quarter":"Q3","year":"2024"},
            "NVDA_Q2_2025":  {"company":"NVIDIA",    "ticker":"NVDA",  "quarter":"Q2","year":"2025"},
            "META_Q3_2024":  {"company":"Meta",      "ticker":"META",  "quarter":"Q3","year":"2024"},
        }
        for doc_id, text in SAMPLE_TRANSCRIPTS.items():
            chunks = splitter.split_text(text.strip())
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                m = meta_map.get(doc_id, {})
                all_metadata.append({**m, "doc_id": doc_id, "chunk_id": f"{doc_id}_{i}"})
        vectorstore = Chroma.from_texts(
            texts=all_chunks, embedding=embedding_model,
            metadatas=all_metadata, persist_directory=CHROMA_DIR,
            collection_name="earnings_calls"
        )
    else:
        vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embedding_model,
            collection_name="earnings_calls"
        )
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=LLM_MODEL,
        temperature=0.1,
        max_tokens=600
    )
    return vectorstore, llm

@st.cache_data
def load_data():
    # Use hardcoded data — no SQLite needed for deployment
    df = pd.DataFrame([
        {"id":1,"doc_id":"AAPL_Q4_2024", "company":"Apple",     "ticker":"AAPL",  "quarter":"Q4","year":2024,"revenue_bn":94.9, "yoy_growth":6.0,   "key_topics":"iPhone,Services,AI",       "n_chunks":12},
        {"id":2,"doc_id":"MSFT_Q1_2025", "company":"Microsoft", "ticker":"MSFT",  "quarter":"Q1","year":2025,"revenue_bn":65.6, "yoy_growth":16.0,  "key_topics":"Azure,AI,Copilot,Cloud",   "n_chunks":10},
        {"id":3,"doc_id":"GOOGL_Q3_2024","company":"Google",    "ticker":"GOOGL", "quarter":"Q3","year":2024,"revenue_bn":88.3, "yoy_growth":15.0,  "key_topics":"Search,YouTube,Gemini,AI", "n_chunks":11},
        {"id":4,"doc_id":"NVDA_Q2_2025", "company":"NVIDIA",    "ticker":"NVDA",  "quarter":"Q2","year":2025,"revenue_bn":30.0, "yoy_growth":122.0, "key_topics":"Data Center,Blackwell,AI", "n_chunks":9},
        {"id":5,"doc_id":"META_Q3_2024", "company":"Meta",      "ticker":"META",  "quarter":"Q3","year":2024,"revenue_bn":40.6, "yoy_growth":19.0,  "key_topics":"Advertising,AI,Llama",     "n_chunks":10},
    ])
    eval_summary = {}
    if os.path.exists(f"{OUTPUT_DIR}/eval_summary.json"):
        with open(f"{OUTPUT_DIR}/eval_summary.json") as f:
            eval_summary = json.load(f)
    eval_results = pd.DataFrame()
    if os.path.exists(f"{OUTPUT_DIR}/eval_results.csv"):
        eval_results = pd.read_csv(f"{OUTPUT_DIR}/eval_results.csv")
    return df, eval_summary, eval_results

# ── RAG function ──────────────────────────────────────────────
def ask_rag(question, company_filter=None, chat_history=None):
    if company_filter and company_filter != "All Companies":
        docs = vectorstore.similarity_search(
            question, k=TOP_K, filter={"company": company_filter}
        )
    else:
        docs = vectorstore.similarity_search(question, k=TOP_K)

    context = "\n\n---\n\n".join([
        f"[{d.metadata.get('company')} {d.metadata.get('quarter')} {d.metadata.get('year')}]\n{d.page_content}"
        for d in docs
    ])
    sources = list(set([
        f"{d.metadata.get('company')} {d.metadata.get('quarter')} {d.metadata.get('year')}"
        for d in docs
    ]))

    # Build conversation history context
    history_context = ""
    if chat_history:
        recent = chat_history[-3:]  # Last 3 exchanges
        history_context = "\n".join([
            f"Previous Q: {h['question']}\nPrevious A: {h['answer'][:200]}"
            for h in recent
        ])

    system_prompt = """You are a financial analyst assistant. Answer questions based ONLY on the provided earnings call transcripts.
Rules:
1. Only use information from the provided context
2. If the answer is not in the context, say "This information is not available in the provided transcripts"
3. Always cite which company and quarter your answer comes from
4. Be concise, factual, and use bullet points for multi-part answers
5. Never make up numbers or facts"""

    user_prompt = f"""{"Previous conversation:\n" + history_context + "\n\n" if history_context else ""}Context from earnings call transcripts:
{context}

Question: {question}

Answer based only on the context above:"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])
    answer = response.content

    # Confidence + grounding check
    check_response = llm.invoke([HumanMessage(content=f"""
Given this context:
{context[:1000]}

And this answer:
{answer}

1. Is the answer fully supported by the context? (YES/NO)
2. Confidence score 0-100 based on how much evidence exists in context.
Reply in exactly this format:
GROUNDED: YES
CONFIDENCE: 85""")])

    check_text = check_response.content.upper()
    is_grounded = "GROUNDED: YES" in check_text
    try:
        conf_line = [l for l in check_text.split("\n") if "CONFIDENCE:" in l][0]
        confidence = int(''.join(filter(str.isdigit, conf_line)))
        confidence = min(max(confidence, 0), 100)
    except:
        confidence = 80 if is_grounded else 40

    return {
        "answer":      answer,
        "sources":     sources,
        "chunks":      docs,
        "is_grounded": is_grounded,
        "confidence":  confidence,
        "context":     context,
    }

# ── Hero Banner ───────────────────────────────────────────────
n_chunks = vectorstore._collection.count()
st.markdown(f"""
<div class="hero-banner">
  <div class="hero-title">📈 Earnings Call Analyst</div>
  <div class="hero-subtitle">Chat with {len(transcripts_df)} company earnings calls · {n_chunks} indexed chunks · Llama 3.3 70B + RAG</div>
  <div style="margin-top:14px;">
    <span class="badge badge-purple">🤖 Llama 3.3 70B</span>
    <span class="badge badge-blue">💬 Chat Memory</span>
    <span class="badge badge-green">⚖️ Compare Mode</span>
    <span class="badge badge-orange">🎯 Confidence Scores</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── KPI Row ───────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.markdown(f"""<div class="metric-card"><div class="metric-value">{len(transcripts_df)}</div><div class="metric-label">Companies</div></div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""<div class="metric-card"><div class="metric-value">{n_chunks}</div><div class="metric-label">Chunks Indexed</div></div>""", unsafe_allow_html=True)
with k3:
    gr = eval_summary.get("grounding_rate", 0)
    st.markdown(f"""<div class="metric-card"><div class="metric-value">{gr:.0f}%</div><div class="metric-label">Grounding Rate</div></div>""", unsafe_allow_html=True)
with k4:
    st.markdown(f"""<div class="metric-card"><div class="metric-value">{len(st.session_state.chat_history)}</div><div class="metric-label">Questions Asked</div></div>""", unsafe_allow_html=True)
with k5:
    avg_conf = np.mean([h.get("confidence", 80) for h in st.session_state.chat_history]) if st.session_state.chat_history else 0
    st.markdown(f"""<div class="metric-card"><div class="metric-value">{avg_conf:.0f}%</div><div class="metric-label">Avg Confidence</div></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💬  Chat", "⚖️  Compare", "📊  Overview", "🧪  Evaluation", "🗄️  SQL"
])

# ─────────────────────────────────────────────────────────────
# TAB 1: CHAT
# ─────────────────────────────────────────────────────────────
with tab1:
    # Sidebar-style controls inside tab
    col_input, col_filter = st.columns([4, 1])
    with col_filter:
        company_filter = st.selectbox("Company",
                                       ["All Companies"] + transcripts_df["company"].tolist(),
                                       label_visibility="collapsed")

    # Suggested questions
    suggested = [
        "What was NVIDIA's revenue growth?",
        "How is Microsoft monetizing AI?",
        "What did Google say about AI Search?",
        "Compare AI CapEx across companies",
        "What is Apple's AI strategy?",
    ]
    st.markdown("**💡 Suggested questions:**")
    s_cols = st.columns(5)
    clicked_q = None
    for i, (col, q) in enumerate(zip(s_cols, suggested)):
        if col.button(q[:30] + "...", key=f"s_{i}", use_container_width=True):
            clicked_q = q

    # Chat input
    with col_input:
        question = st.chat_input("Ask anything about the earnings calls...")

    if clicked_q:
        question = clicked_q

    # Display chat history
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            # User bubble
            st.markdown(f'<div class="user-bubble">🙋 {msg["question"]}</div>', unsafe_allow_html=True)

            # Bot bubble
            conf  = msg.get("confidence", 80)
            conf_color = "#4ADE80" if conf >= 80 else "#FBBF24" if conf >= 60 else "#F87171"
            grounded_text = "✅ Grounded" if msg.get("is_grounded") else "⚠️ Unverified"

            sources_html = "".join([f'<span class="source-tag">{s}</span>' for s in msg.get("sources", [])])

            st.markdown(f"""
            <div class="bot-bubble">
                <div class="bot-answer">{msg["answer"].replace(chr(10), "<br>").replace("* ", "• ")}</div>
                <div class="bot-meta">
                    <span style="font-size:11px;color:#8B8FA8;">{grounded_text}</span>
                    <span style="font-size:11px;background:{conf_color}22;color:{conf_color};padding:2px 8px;border-radius:10px;border:1px solid {conf_color}44;">
                        Confidence: {conf}%
                    </span>
                    {sources_html}
                </div>
                <div style="margin-top:8px;">
                    <div style="height:3px;border-radius:2px;background:linear-gradient(90deg,{conf_color} {conf}%,#2D2F3E {conf}%);"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Show chunks expander
            with st.expander(f"🔍 View {len(msg.get('chunks', []))} source chunks"):
                for chunk in msg.get("chunks", []):
                    # Highlight keywords from question
                    chunk_text = chunk.page_content
                    for word in msg["question"].split():
                        if len(word) > 4:
                            chunk_text = chunk_text.replace(
                                word, f'<span class="highlight">{word}</span>'
                            )
                    st.markdown(f"""<div class="chunk-card">
                    <strong style="color:#6C63FF;">[{chunk.metadata.get('company')} {chunk.metadata.get('quarter')} {chunk.metadata.get('year')}]</strong><br>
                    {chunk_text}
                    </div>""", unsafe_allow_html=True)

    # Process new question
    if question:
        with st.spinner("🔍 Searching and generating answer..."):
            start = time.time()
            result = ask_rag(question, company_filter, st.session_state.chat_history)
            elapsed = time.time() - start

        # Add to history
        st.session_state.chat_history.append({
            "question":    question,
            "answer":      result["answer"],
            "sources":     result["sources"],
            "chunks":      result["chunks"],
            "is_grounded": result["is_grounded"],
            "confidence":  result["confidence"],
            "elapsed":     elapsed,
        })
        st.rerun()

    # Clear chat button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear conversation", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()

# ─────────────────────────────────────────────────────────────
# TAB 2: COMPARE MODE
# ─────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">Compare Two Companies Side by Side</div>', unsafe_allow_html=True)
    st.markdown("*Ask the same question and see how two companies answered it in their earnings calls*")

    compare_q = st.text_input("Question to compare",
                               placeholder="e.g. What is the company's AI strategy and investment?",
                               value=st.session_state.compare_q)

    c1, c2 = st.columns(2)
    company_a = c1.selectbox("Company A", transcripts_df["company"].tolist(), index=0)
    company_b = c2.selectbox("Company B", transcripts_df["company"].tolist(), index=1)

    if st.button("⚖️ Compare", type="primary") and compare_q:
        st.session_state.compare_q = compare_q

        col_a, col_b = st.columns(2)

        with col_a:
            with st.spinner(f"Analyzing {company_a}..."):
                result_a = ask_rag(compare_q, company_a)
            conf_a = result_a["confidence"]
            conf_color_a = "#4ADE80" if conf_a >= 80 else "#FBBF24" if conf_a >= 60 else "#F87171"
            st.markdown(f"""<div class="compare-col">
            <div class="compare-header">📊 {company_a}</div>
            <div style="font-size:13px;color:#E0E3F0;line-height:1.7;">{result_a["answer"].replace(chr(10),"<br>").replace("* ","• ")}</div>
            <div style="margin-top:12px;">
                <span style="font-size:11px;background:{conf_color_a}22;color:{conf_color_a};padding:2px 8px;border-radius:10px;border:1px solid {conf_color_a}44;">
                    Confidence: {conf_a}%
                </span>
                {'<span style="font-size:11px;color:#4ADE80;margin-left:8px;">✅ Grounded</span>' if result_a["is_grounded"] else '<span style="font-size:11px;color:#F87171;margin-left:8px;">⚠️ Unverified</span>'}
            </div>
            <div style="margin-top:8px;height:3px;border-radius:2px;background:linear-gradient(90deg,{conf_color_a} {conf_a}%,#2D2F3E {conf_a}%);"></div>
            </div>""", unsafe_allow_html=True)

        with col_b:
            with st.spinner(f"Analyzing {company_b}..."):
                result_b = ask_rag(compare_q, company_b)
            conf_b = result_b["confidence"]
            conf_color_b = "#4ADE80" if conf_b >= 80 else "#FBBF24" if conf_b >= 60 else "#F87171"
            st.markdown(f"""<div class="compare-col">
            <div class="compare-header">📊 {company_b}</div>
            <div style="font-size:13px;color:#E0E3F0;line-height:1.7;">{result_b["answer"].replace(chr(10),"<br>").replace("* ","• ")}</div>
            <div style="margin-top:12px;">
                <span style="font-size:11px;background:{conf_color_b}22;color:{conf_color_b};padding:2px 8px;border-radius:10px;border:1px solid {conf_color_b}44;">
                    Confidence: {conf_b}%
                </span>
                {'<span style="font-size:11px;color:#4ADE80;margin-left:8px;">✅ Grounded</span>' if result_b["is_grounded"] else '<span style="font-size:11px;color:#F87171;margin-left:8px;">⚠️ Unverified</span>'}
            </div>
            <div style="margin-top:8px;height:3px;border-radius:2px;background:linear-gradient(90deg,{conf_color_b} {conf_b}%,#2D2F3E {conf_b}%);"></div>
            </div>""", unsafe_allow_html=True)

        # Confidence comparison chart
        st.markdown("<br>", unsafe_allow_html=True)
        fig_conf = go.Figure(go.Bar(
            x=[company_a, company_b],
            y=[result_a["confidence"], result_b["confidence"]],
            marker_color=[COLORS["primary"], COLORS["secondary"]],
            text=[f"{result_a['confidence']}%", f"{result_b['confidence']}%"],
            textposition="outside", textfont=dict(color="white", size=14)
        ))
        fig_conf.add_hline(y=80, line_color=COLORS["success"], line_dash="dash",
                           line_width=1.5, annotation_text="High confidence threshold")
        fig_conf.update_layout(
            template=PLOT_TEMPLATE, height=280,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(26,29,39,0.5)",
            yaxis=dict(title="Confidence Score (%)", range=[0,110], gridcolor="#2D2F3E"),
            xaxis=dict(gridcolor="#2D2F3E"),
            title=f"Answer Confidence: {company_a} vs {company_b}",
            showlegend=False
        )
        st.plotly_chart(fig_conf, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# TAB 3: COMPANY OVERVIEW
# ─────────────────────────────────────────────────────────────
with tab3:
    col1, col2 = st.columns(2)
    with col1:
        fig_rev = go.Figure(go.Bar(
            x=transcripts_df["company"], y=transcripts_df["revenue_bn"],
            marker=dict(color=transcripts_df["revenue_bn"],
                        colorscale=[[0,"#1A2744"],[1,"#6C63FF"]], opacity=0.9),
            text=transcripts_df["revenue_bn"].apply(lambda x: f"${x}B"),
            textposition="outside", textfont=dict(color="white")
        ))
        fig_rev.update_layout(template=PLOT_TEMPLATE, height=350,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(26,29,39,0.5)",
            title="Revenue by Company ($B)", yaxis=dict(gridcolor="#2D2F3E"),
            xaxis=dict(gridcolor="#2D2F3E"), showlegend=False)
        st.plotly_chart(fig_rev, use_container_width=True)

    with col2:
        colors_g = [COLORS["success"] if v > 20 else COLORS["primary"] for v in transcripts_df["yoy_growth"]]
        fig_g = go.Figure(go.Bar(
            x=transcripts_df["company"], y=transcripts_df["yoy_growth"],
            marker_color=colors_g, opacity=0.9,
            text=transcripts_df["yoy_growth"].apply(lambda x: f"+{x}%"),
            textposition="outside", textfont=dict(color="white")
        ))
        fig_g.update_layout(template=PLOT_TEMPLATE, height=350,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(26,29,39,0.5)",
            title="YoY Revenue Growth %", yaxis=dict(gridcolor="#2D2F3E"),
            xaxis=dict(gridcolor="#2D2F3E"), showlegend=False)
        st.plotly_chart(fig_g, use_container_width=True)

    # AI mentions
    ai_counts = []
    for _, row in transcripts_df.iterrows():
        doc_id = f"{row['ticker']}_{row['quarter']}_{row['year']}"
        transcript = SAMPLE_TRANSCRIPTS.get(doc_id, "")
        ai_count = transcript.lower().count("ai") + transcript.lower().count("artificial intelligence")
        ai_counts.append({"company": row["company"], "ai_mentions": ai_count})
    ai_df = pd.DataFrame(ai_counts).sort_values("ai_mentions", ascending=True)

    fig_ai = go.Figure(go.Bar(
        x=ai_df["ai_mentions"], y=ai_df["company"], orientation="h",
        marker=dict(color=ai_df["ai_mentions"], colorscale=[[0,"#1A2744"],[1,"#6C63FF"]]),
        text=ai_df["ai_mentions"], textposition="outside", textfont=dict(color="white")
    ))
    fig_ai.update_layout(template=PLOT_TEMPLATE, height=300,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(26,29,39,0.5)",
        title="AI Mentions in Earnings Calls",
        xaxis=dict(title="# Mentions", gridcolor="#2D2F3E"),
        yaxis=dict(gridcolor="#2D2F3E"), showlegend=False)
    st.plotly_chart(fig_ai, use_container_width=True)

    st.dataframe(transcripts_df[["company","ticker","quarter","year","revenue_bn","yoy_growth","key_topics","n_chunks"]],
                 use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────
# TAB 4: EVALUATION
# ─────────────────────────────────────────────────────────────
with tab4:
    col1, col2 = st.columns(2)
    with col1:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=eval_summary.get("grounding_rate", 0),
            title={"text": "Grounding Rate (%)"},
            gauge={"axis": {"range": [0,100]}, "bar": {"color": COLORS["primary"]},
                   "bgcolor": "#1A1D27",
                   "steps": [{"range":[0,60],"color":"#2A1A1A"},{"range":[60,80],"color":"#2A2A1A"},{"range":[80,100],"color":"#1A2A1A"}],
                   "threshold": {"line": {"color": COLORS["success"], "width": 3}, "thickness": 0.75, "value": 80}},
            number={"suffix": "%", "font": {"color": "#E0E3F0"}}
        ))
        fig_gauge.update_layout(template=PLOT_TEMPLATE, height=280, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col2:
        if len(eval_results) > 0:
            cat_s = eval_results.groupby("category").agg(
                n=("question","count"),
                src=("source_correct","mean"),
                grd=("is_grounded","mean")
            ).reset_index()
            cat_s["src"] = (cat_s["src"]*100).round(1).astype(str) + "%"
            cat_s["grd"] = (cat_s["grd"]*100).round(1).astype(str) + "%"
            cat_s.columns = ["Category","Questions","Source Acc","Grounding"]
            st.dataframe(cat_s, use_container_width=True, hide_index=True)

    if len(eval_results) > 0:
        display_eval = eval_results[["question","category","source_correct","is_grounded","answer"]].copy()
        display_eval["source_correct"] = display_eval["source_correct"].map({True:"✅",False:"❌"})
        display_eval["is_grounded"]    = display_eval["is_grounded"].map({True:"✅",False:"❌"})
        display_eval["answer"]         = display_eval["answer"].str[:100] + "..."
        display_eval.columns = ["Question","Category","Correct Source","Grounded","Answer Preview"]
        st.dataframe(display_eval, use_container_width=True, hide_index=True)

    # Chat session stats
    if st.session_state.chat_history:
        st.markdown('<div class="section-header">This Session</div>', unsafe_allow_html=True)
        session_df = pd.DataFrame([{
            "Question": h["question"][:60] + "...",
            "Confidence": f"{h.get('confidence',80)}%",
            "Grounded": "✅" if h.get("is_grounded") else "⚠️",
            "Response Time": f"{h.get('elapsed',0):.1f}s"
        } for h in st.session_state.chat_history])
        st.dataframe(session_df, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────
# TAB 5: SQL EXPLORER
# ─────────────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-header">SQL Metadata Explorer</div>', unsafe_allow_html=True)
    if not os.path.exists(DB_PATH):
        load_data()
    conn_sql = sqlite3.connect(DB_PATH)

    queries = {
        "All Transcripts": "SELECT * FROM transcripts ORDER BY yoy_growth DESC",
        "High Growth (>15%)": "SELECT company, ticker, quarter, year, revenue_bn, yoy_growth FROM transcripts WHERE yoy_growth > 15 ORDER BY yoy_growth DESC",
        "AI Companies": "SELECT company, ticker, key_topics, n_chunks FROM transcripts WHERE key_topics LIKE '%AI%'",
        "Revenue Summary": "SELECT company, SUM(revenue_bn) as revenue, AVG(yoy_growth) as avg_growth FROM transcripts GROUP BY company ORDER BY revenue DESC",
        "Custom Query": "SELECT * FROM transcripts LIMIT 5"
    }

    sel = st.selectbox("Query template", list(queries.keys()))
    q_t = queries[sel]
    if sel == "Custom Query":
        q_t = st.text_area("SQL Editor", value=q_t, height=100)
    else:
        st.code(q_t, language="sql")

    if st.button("▶ Run Query", type="primary"):
        try:
            res = pd.read_sql_query(q_t, conn_sql)
            st.dataframe(res, use_container_width=True, hide_index=True)
            num_cols = res.select_dtypes(include=np.number).columns.tolist()
            cat_cols = res.select_dtypes(exclude=np.number).columns.tolist()
            if num_cols and cat_cols:
                fig = px.bar(res, x=cat_cols[0], y=num_cols[0],
                              color_discrete_sequence=[COLORS["primary"]],
                              template=PLOT_TEMPLATE)
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                   plot_bgcolor="rgba(26,29,39,0.5)", height=300,
                                   xaxis=dict(gridcolor="#2D2F3E"),
                                   yaxis=dict(gridcolor="#2D2F3E"))
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")
    conn_sql.close()

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#4A4E6A;font-size:12px;'>"
    "Earnings Call RAG · Llama 3.3 70B via Groq · ChromaDB · LangChain · SQL Metadata · Chat Memory"
    "</div>", unsafe_allow_html=True
)
