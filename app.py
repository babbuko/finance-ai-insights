"""
Finance Insights Web Application
Flask backend coordinating Traditional ML Sentiment Classification & LLM Insights.
"""

import os
import requests
from flask import Flask, render_template, request
from model_training import load_sentiment_pipeline, train_sentiment_model

app = Flask(__name__)

if not (os.path.exists("sentiment_model.pkl") and os.path.exists("vectorizer.pkl")):
    print("[!] Model artifacts not detected. Running initial model training...")
    train_sentiment_model()

predict_sentiment = load_sentiment_pipeline()

OLLAMA_API_URL = "http://localhost:11434/api/generate"
LM_STUDIO_API_URL = "http://localhost:1234/v1/chat/completions"


def query_llm(prompt: str) -> str:
    """Queries local Ollama or LM Studio, with a rule-based fallback."""
    try:
        payload = {"model": "llama3", "prompt": prompt, "stream": False}
        res = requests.post(OLLAMA_API_URL, json=payload, timeout=4)
        if res.status_code == 200:
            return res.json().get("response", "").strip()
    except Exception:
        pass

    try:
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3
        }
        res = requests.post(LM_STUDIO_API_URL, json=payload, timeout=4)
        if res.status_code == 200:
            return res.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        pass

    return (
        "• Executive Summary: Reported financial events indicate material business impact.\n"
        "• Primary Drivers: Direct shifts in quarterly operational performance, guidance, or regulatory posture.\n"
        "• Disclaimer: Synthesized via fallback engine; activate Ollama or LM Studio for dynamic generative inference."
    )


def generate_llm_summary(text: str) -> str:
    prompt = (
        "You are an expert Wall Street financial analyst. Summarize the following news "
        "or earnings report into 3 clear, analytical bullet points. Do not provide trading advice.\n\n"
        f"Text:\n{text}\n\nSummary:"
    )
    return query_llm(prompt)


def generate_llm_qa(text: str, question: str) -> str:
    prompt = (
        "You are an investment analyst. Based strictly on the provided financial text, "
        "answer the user's specific question concisely with supporting context.\n\n"
        f"Financial Text:\n{text}\n\n"
        f"User Question: {question}\n\nAnswer:"
    )
    return query_llm(prompt)


@app.route("/", methods=["GET", "POST"])
def index():
    user_text = ""
    user_query = ""
    sentiment_result = None
    llm_summary = None
    llm_qa_response = None

    if request.method == "POST":
        user_text = request.form.get("user_text", "").strip()
        user_query = request.form.get("user_query", "").strip()

        if user_text:
            sentiment_result = predict_sentiment(user_text)
            llm_summary = generate_llm_summary(user_text)

            if user_query:
                llm_qa_response = generate_llm_qa(user_text, user_query)

    return render_template(
        "index.html",
        user_text=user_text,
        user_query=user_query,
        sentiment_result=sentiment_result,
        llm_summary=llm_summary,
        llm_qa_response=llm_qa_response
    )


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
