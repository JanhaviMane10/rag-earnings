# ============================================================
# config.py — Central settings for the RAG project
# ============================================================

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- API Keys ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Paths ---
DATA_DIR    = "data"
OUTPUT_DIR  = "outputs"
DB_PATH     = "outputs/earnings.db"
CHROMA_DIR  = "outputs/chroma_db"

for folder in [DATA_DIR, OUTPUT_DIR]:
    os.makedirs(folder, exist_ok=True)

# --- LLM Settings ---
LLM_MODEL = "llama-3.3-70b-versatile"  # Groq's free Llama 3.1 70B
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Free local embedding model
CHUNK_SIZE      = 500                  # Characters per chunk
CHUNK_OVERLAP   = 50                   # Overlap between chunks
TOP_K           = 5                    # Number of chunks to retrieve

# --- Companies to analyze ---
# These are well-known companies with public earnings transcripts
COMPANIES = [
    {"name": "Apple",     "ticker": "AAPL"},
    {"name": "Microsoft", "ticker": "MSFT"},
    {"name": "Google",    "ticker": "GOOGL"},
    {"name": "Amazon",    "ticker": "AMZN"},
    {"name": "Meta",      "ticker": "META"},
    {"name": "Netflix",   "ticker": "NFLX"},
    {"name": "Tesla",     "ticker": "TSLA"},
    {"name": "NVIDIA",    "ticker": "NVDA"},
]

# --- Sample earnings transcripts (we'll use these as demo data) ---
# In a real project you'd scrape from SEC EDGAR
SAMPLE_TRANSCRIPTS = {
    "AAPL_Q4_2024": """
Apple Inc. Q4 2024 Earnings Call Transcript

CEO Tim Cook: Good afternoon everyone. We're pleased to report another strong quarter.
Revenue came in at $94.9 billion, up 6% year over year. iPhone revenue was $46.2 billion,
up 6% from the year-ago quarter. Services revenue reached an all-time high of $24.2 billion,
up 12% year over year. We now have over 1 billion paid subscriptions across our services.

On AI, we're incredibly excited about Apple Intelligence. We believe we're at an inflection
point where AI is fundamentally changing how people interact with their devices. Apple
Intelligence will be available on iPhone 16, iPhone 15 Pro, and compatible iPad and Mac
devices. We're rolling it out in US English first, with more languages coming next year.

CFO Luca Maestri: Gross margin was 46.2%, up 100 basis points year over year. Operating
cash flow was $26.8 billion. We returned $29 billion to shareholders during the quarter.
For Q1 2025, we expect revenue between $89 billion and $93 billion.

Analyst: Can you discuss the AI monetization strategy?
Tim Cook: We see Apple Intelligence as a way to increase the value of our ecosystem.
While it's included with devices initially, we believe it will drive higher device sales
and services engagement over time. We're very early in understanding the full monetization
potential but are extremely optimistic about the opportunity.
""",

    "MSFT_Q1_2025": """
Microsoft Corporation Q1 2025 Earnings Call Transcript

CEO Satya Nadella: Thank you. We had an outstanding quarter with revenue of $65.6 billion,
up 16% year over year. Our commercial cloud revenue was $38.9 billion, up 22%.

Azure and other cloud services grew 33% in constant currency. AI services are becoming
a meaningful contributor to Azure growth. GitHub Copilot now has 1.8 million paid subscribers,
up 55% year over year. Microsoft 365 Copilot is being adopted by 70% of Fortune 500 companies.

CFO Amy Hood: Operating income was $30.6 billion, up 14%. We generated $34.2 billion
in operating cash flow. Capital expenditure was $20 billion, primarily for cloud and AI
infrastructure. We expect Q2 revenue between $68.1 billion and $69.1 billion.

Analyst: How should we think about AI contribution to Azure growth?
Satya Nadella: AI is now contributing roughly 12 percentage points to Azure growth.
We're seeing customers move from experimentation to production workloads at an
accelerating pace. The ROI story for Copilot is becoming very clear for customers.
""",

    "GOOGL_Q3_2024": """
Alphabet Inc. Q3 2024 Earnings Call Transcript

CEO Sundar Pichai: We had a very strong quarter. Total revenues were $88.3 billion,
up 15% year over year. Google Search revenues grew 12% to $49.4 billion. YouTube
advertising revenues were $8.9 billion, up 12%. Google Cloud revenues grew 35% to
$11.4 billion, with operating profit of $1.9 billion.

On AI, Gemini is being used across all our products. Google Search now incorporates
AI Overviews reaching over 1 billion users. We're seeing strong Cloud growth driven
by AI infrastructure demand and our Vertex AI platform.

CFO Philipp Schindler: Operating margin expanded to 28%. We returned $15.3 billion
to shareholders. CapEx was $13.1 billion for the quarter, focused on technical
infrastructure including AI compute.

Analyst: Can you discuss Search monetization with AI Overviews?
Sundar Pichai: We're pleased with how AI Overviews are performing. We're seeing
higher user satisfaction scores and strong advertiser results. The monetization rate
is in line with traditional Search, and we see a long runway to improve further.
""",

    "NVDA_Q2_2025": """
NVIDIA Corporation Q2 FY2025 Earnings Call Transcript

CEO Jensen Huang: Revenue was a record $30.0 billion, up 122% year over year and
up 15% sequentially. Data Center revenue was a record $26.3 billion, up 154% year
over year. Our Blackwell architecture is in full production and demand exceeds supply.

We are at the beginning of a new industrial revolution. Every country, every company
needs sovereign AI infrastructure. The total addressable market for AI compute,
networking, and software is measured in trillions of dollars.

CFO Colette Kress: Gross margin was 75.1%, up from 70.1% a year ago. We generated
$14.5 billion in free cash flow. We returned $7.8 billion to shareholders. For Q3,
we expect revenue of approximately $32.5 billion.

Analyst: How should we think about Blackwell supply constraints?
Jensen Huang: Blackwell demand is extraordinary. We're working with our supply chain
partners to increase production as fast as possible. We expect Blackwell to be a
significant contributor to revenue in Q3 and ramp substantially through fiscal 2026.
The transition from Hopper to Blackwell is going extremely well.
""",

    "META_Q3_2024": """
Meta Platforms Q3 2024 Earnings Call Transcript

CEO Mark Zuckerberg: Q3 was a great quarter. Revenue was $40.6 billion, up 19%
year over year. Daily active people across our family of apps reached 3.29 billion,
up 5% year over year. We continue to see strong engagement across Facebook, Instagram,
WhatsApp, and Threads.

On AI, Meta AI now has over 500 million monthly active users across our apps.
Llama models have been downloaded over 400 million times. We're seeing AI drive
meaningful improvements in ad performance and content recommendations.

CFO Susan Li: Operating income was $17.3 billion, representing a 43% margin.
Capital expenditures were $9.2 billion. We expect Q4 revenue between $45 and $48
billion. Full year 2024 CapEx expected to be $38-40 billion, with significant
increases in 2025 for AI infrastructure.

Analyst: Can you discuss the ROI on AI infrastructure spending?
Mark Zuckerberg: We're seeing AI deliver real business results. Our AI-driven ad
targeting improvements have increased advertiser ROI significantly. Recommendation
AI has increased time spent on our platforms by meaningful amounts. We believe
the infrastructure investment will pay off substantially over the next several years.
""",
}
