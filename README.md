# Multi-Document RAG Chatbot with Source Retrieval

## Overview
This project implements a Retrieval-Augmented Generation (RAG) system that retrieves relevant information from multiple documents and generates answers with source references.

## Features
- Multi-document search
- Semantic retrieval using embeddings
- Source attribution for answers
- Scalable architecture

## How it works
1. Documents are split into chunks
2. Each chunk is converted into embeddings
3. Stored in vector database (FAISS)
4. Query → embedding → similarity search
5. Top results are combined to generate answer

## Tech Stack
- Python
- Sentence Transformers
- FAISS (Vector Database)

## Endee Integration
FAISS is used for local prototyping. In a production system, Endee vector database can replace FAISS to provide:
- Scalable vector storage
- Faster retrieval
- Filtering and metadata support

## Future Improvements
- Integrate LLM (OpenAI) for better answers
- Build UI (Streamlit)
- Deploy as API
