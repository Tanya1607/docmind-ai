# DocMind AI

**AI-powered PDF document intelligence using Retrieval-Augmented Generation (RAG).**

Upload PDF documents, ask questions about their contents, retrieve relevant passages, and generate document insights with page-level evidence.

## Features

- Upload one or more text-based PDFs
- Extract page-level text and split it into passages
- Generate local sentence embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- Retrieve relevant passages with cosine similarity
- Ask evidence-grounded questions with source references
- Generate executive summaries, key findings, action items, and important dates
- Keep uploaded document text and embeddings in the current app session

## Tech stack

Python · Streamlit · Groq API · Sentence Transformers · NumPy · PyPDF · RAG

## Run locally

1. Install Python 3.10 or later.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Create an API key at [Groq Console](https://console.groq.com/keys).
4. Set the environment variable in Windows PowerShell:

   ```powershell
   $env:GROQ_API_KEY="your-groq-api-key"
   ```

5. Start the app:

   ```bash
   streamlit run app.py
   ```

## Deploy on Streamlit Community Cloud

1. Push `app.py`, `requirements.txt`, and `README.md` to a GitHub repository.
2. Create an app at [Streamlit Community Cloud](https://share.streamlit.io/) and select this repository, the `main` branch, and `app.py`.
3. In the app's **Settings → Secrets**, add:

   ```toml
   GROQ_API_KEY = "your-groq-api-key"
   GROQ_MODEL = "openai/gpt-oss-20b"
   ```

4. Deploy and test with a text-based PDF.

Never commit API keys or `secrets.toml` to GitHub. Groq model availability and usage limits may change; check the [Groq supported models](https://console.groq.com/docs/models) page if a model ID is unavailable.

## Architecture

1. PDF ingestion and page-level text extraction
2. Text chunking with overlap
3. Sentence-transformer embeddings
4. Similarity-based retrieval
5. Hosted LLM response generation using retrieved evidence

## Limitations

- Scanned PDFs need OCR, which is not included.
- Answer quality depends on the PDF content, retrieval quality, and selected model.
- Hosted model requests are subject to the provider's availability, usage limits, and terms.
- Uploaded files and embeddings are not stored in a persistent database.

## Future improvements

- Multi-format document support
- Retrieval and answer quality evaluation
- Persistent document storage and user workspaces

## Author

Built as a Generative AI demonstration project.
