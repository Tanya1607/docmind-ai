# DocMind AI

A deployable GenAI document intelligence demo featuring PDF ingestion, embeddings-based retrieval (RAG), source-grounded Q&A, page references, and generated document insights.

## Features
- Upload one or more text-based PDFs
- Extract page-level text and chunk it
- Retrieve relevant passages with OpenAI embeddings
- Ask questions and receive answers grounded in retrieved passages
- Generate executive summaries, key findings, action items, and important dates
- Download generated insights as text
- No document database: uploaded content and embeddings stay in the app session

## Run locally
1. Install Python 3.10+.
2. Create and activate a virtual environment.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set your API key:
   - macOS/Linux: `export OPENAI_API_KEY="your-key"`
   - Windows PowerShell: `$env:OPENAI_API_KEY="your-key"`
5. Start the app:
   ```bash
   streamlit run app.py
   ```

## Deploy on Streamlit Community Cloud
1. Push `app.py`, `requirements.txt`, and `README.md` to a GitHub repository.
2. Create a new app at https://share.streamlit.io/ and select the repository and `app.py`.
3. In app settings, add a secret:
   ```toml
   OPENAI_API_KEY = "your-key"
   ```
4. Deploy, open the public URL in a private/incognito window, and test with a sample PDF.

## Suggested live demo
1. Upload a short research paper, annual report, or project brief.
2. Ask: "What are the three main findings? Cite the pages."
3. Ask a detail-specific question whose answer is present on a page near the end.
4. Open the evidence panel and show the extracted passages.
5. Generate an executive summary or action items.

## Current limitations
- Text-based PDFs only; scanned PDFs need OCR.
- The insight generator uses a bounded set of passages to keep the MVP fast.
- Similarity scores are retrieval signals, not confidence probabilities.
- Review generated output against the source document before relying on it.
