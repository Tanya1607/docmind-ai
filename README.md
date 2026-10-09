# DocMind AI

**AI-powered PDF Document Intelligence using Retrieval-Augmented Generation (RAG).**

DocMind AI allows users to upload PDF documents, ask questions about their contents, retrieve relevant text passages, and generate document insights using locally hosted language models.

## Features

- Upload and process PDF documents.
- Extract text and split it into searchable chunks.
- Generate embeddings using `nomic-embed-text`.
- Retrieve relevant document passages using semantic similarity.
- Answer questions using `llama3.2:3b`.
- Generate insights from uploaded documents.
- Display source-based responses to help users verify answers.
- Run locally using Ollama without requiring an OpenAI API key.

## Tech Stack

- Python
- Streamlit
- Ollama
- Llama 3.2
- Nomic Embed Text
- Retrieval-Augmented Generation (RAG)
- NumPy
- PyPDF

## Run Locally

1. Install Python 3.10 or later.
2. Install [Ollama](https://ollama.com/).
3. Download the required models:

   ```bash
   ollama pull llama3.2:3b
   ollama pull nomic-embed-text
   ```

4. Install the project dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Start the application:

   ```bash
   streamlit run app.py
   ```

Make sure Ollama is running before using the application.

## Architecture

1. **PDF ingestion:** Extract text from uploaded documents.
2. **Chunking:** Divide the extracted text into smaller passages.
3. **Embedding generation:** Convert passages into numerical vectors.
4. **Retrieval:** Find relevant passages using semantic similarity.
5. **Answer generation:** Generate answers grounded in retrieved document content.

## Limitations

- Answer quality depends on the document content and the selected models.
- Scanned PDFs may require OCR if they do not contain extractable text.
- Models must be installed locally through Ollama.
- Uploaded documents and embeddings are handled within the application session.

## Future Improvements

- Cloud deployment with hosted inference.
- Improved document citation and retrieval evaluation.
- Support for additional document formats.
- Persistent document storage and multi-document search.

## Author

Developed as a Generative AI project demonstrating PDF processing, semantic search, and RAG-based question answering.
