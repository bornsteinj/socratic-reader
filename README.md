# Socratic Dialogue Chatbot (Streamlit)

A minimal, FERPA-friendly Socratic reading coach for graduate courses.

## Features
- One-question-at-a-time Socratic loop
- Page-anchored reasoning (optionally required)
- Equity prompts embedded throughout
- PDF upload + text extraction for page previews
- 4-part wrap-up export to DOCX
- No server-side logging (session-only state)

## Quick Start (Streamlit Community Cloud)
1. **Create a new app** from this repo (fork or upload the files).  
2. **Add the secret** `OPENAI_API_KEY` in *App settings → Secrets* (or set as an env var on your own server).  
3. Set **Main file** to `app.py`.  
4. Deploy. Open the URL and share it in your LMS as an external link.

## Local dev
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
streamlit run app.py