# Semantic Quote Search Assistant (RAG-powered)

This project implements a **semantic quote retrieval system** using **Retrieval-Augmented Generation (RAG)** on the [Abirate/english_quotes](https://huggingface.co/datasets/Abirate/english_quotes) dataset. It allows users to enter natural-language queries (e.g., *"quotes about hope by Oscar Wilde"*) and retrieves contextually relevant quotes using fine-tuned embeddings and a generative model.

---

## Features

- Semantic quote search with SentenceTransformers
- Fast vector retrieval using FAISS
- Context-aware generation using FLAN-T5 (RAG)
- Evaluation using FLAN-T5 relevance classification
- Interactive Streamlit UI for user queries
- JSON-formatted results with optional download
- Visualizations of author and tag distributions

---

## Project Structure


    ├── data_preparation_and_training.ipynb
    ├── rag_generation_and_evaluation.ipynb
    ├── app.py
    ├── quotes_cleaned.csv inference
    ├── fine-tuned-quote-embedder
    └── README.md



## Installation

    pip install datasets sentence-transformers faiss-cpu transformers accelerate
    pip install arize-phoenix-evals
    pip install streamlit
## Data Preparation
Loaded from HuggingFace: Abirate/english_quotes

- Cleaned quotes and authors with:

- Lowercasing

- Punctuation removal

- Stopword filtering

- Lemmatization (NLTK)

- Converted tags to list objects using ast.literal_eval

- Sampled 2000 unique quotes for efficient fine-tuning

## Model Fine-Tuning
Model: all-MiniLM-L6-v2

- Loss: MultipleNegativesRankingLoss

- Objective: Bring semantically similar quotes closer in embedding space

- Training Data: (quote, quote) pairs (self-supervised)

- Training Epochs: 1

- Output: fine-tuned-quote-embedder/ (saved locally)

## FAISS-Based Retrieval
Indexed quotes + author + tags using FAISS with cosine similarity

- Query embeddings are matched against top-K vectors

- Retrieval time: ~milliseconds for 2000+ entries

## RAG Pipeline
Model: google/flan-t5-base

- Prompt-based generation:

- Used retrieved quotes as context

- Asked model to summarize or answer queries

- Also used for binary relevance classification (prompt: "Is this quote relevant to the query?")

## Evaluation
- Metric	Score
- Precision	1.00
- Recall	0.50
- F1 Score	0.67

- Evaluation done on manually labeled mini dataset using FLAN-T5

- Showed strong precision, indicating high confidence in relevant quotes

- Lower recall due to small sample size and edge-case queries

## Streamlit App
To run the app locally:

    streamlit run app.py

## ⚠️ Limitations
- Limited support for multi-hop queries (e.g., life + love + 20th century)

- FLAN-T5 relevance classification may misinterpret short or abstract quotes

- Evaluation set is small — full validation requires more labeled data

- No metadata filters yet (e.g., filter only Oscar Wilde or only humor quotes)

## Future Work
- Add filters for authors, tags, or time periods

- JSON download button for exported results

- Dynamic feedback loop for improving relevance via user input

- Add optional OpenAI or Llama3 model for better generation

## Acknowledgments

- HuggingFace Datasets

- SentenceTransformers

- FAISS by Facebook AI

- FLAN-T5

- Arize Phoenix

## 📬 Contact
For questions or collaboration:
email - jayanthchatrathi26@gmail.com
