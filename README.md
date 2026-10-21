# Finance AI Insights & Market Sentiment Analyzer

**Course:** ITAI - Artificial Intelligence Applications  
**Module:** AI in Finance (Option 1 Deliverable)  
**Author:** Kamron Robinson  

---

## 1. Project Overview & Business Case

### Business Scenario
Financial analysts and asset managers review high volumes of financial press releases, quarterly disclosures, and regulatory filings each trading day. Manual inspection introduces evaluation latency and subjective sentiment classification.

### Value Proposition
This platform provides a two-tier hybrid intelligence pipeline:
1. **Low-Latency Traditional NLP (Tier 1):** Applies text cleaning, TF-IDF vectorization, and a Multinomial Naive Bayes classifier to tag market sentiment (**positive**, **negative**, or **neutral**) with reproducible, auditable metrics.
2. **Generative LLM Synthesis (Tier 2):** Connects to local runtime APIs (Ollama on `11434` or LM Studio on `1234`) to extract structured bullet-point summaries and answer ad-hoc analytical questions regarding the text.

---

## 2. Technical Architecture
[ Financial Text Input ]
│
├───────────────────────────────┐
▼                               ▼
[ Text Cleaning & TF-IDF ]       [ LLM Prompt Engineering ]
│                               │
▼                               ▼
[ Multinomial Naive Bayes ]       [ Local LLM / Ollama API ]
(Sentiment Classification)        (Summarization & Q&A)
│                               │
└───────────────┬───────────────┘
▼
[ Minimal Flask Interface ]
---

## 3. Installation & Run Guide

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
### Step 2: Train the Sentiment Model
Bash
python model_training.py
Generates sentiment_model.pkl and vectorizer.pkl with cross-validated accuracy.
### Step 3: Run the Application
python app.py
Open your browser and navigate to:
[http://127.0.0.1:5000](http://127.0.0.1:5000)
