# 🏥 RAG for Healthcare – Retrieval-Augmented Generation System

**RAG for Healthcare** is a Retrieval-Augmented Generation (RAG) system designed to answer user queries based on healthcare knowledge extracted from a custom PDF document.

This system allows users to ask questions in natural language and receive accurate answers grounded in medical documents (e.g., stroke/brain hemorrhage information from `dotquy.pdf`).

---

## 🎯 Objectives

- Load a healthcare PDF document (e.g., stroke-related).
- Split the document into meaningful chunks.
- Store and index chunks using a vector store (e.g., FAISS or Qdrant).
- Retrieve relevant content and use an LLM to generate answers.

---

## 🗂️ Project Structure

rag_model/
│
├── venv/ # Global virtual environment (optional)
│
└── rag_healthcare/ # Main application folder
├── .venv/ # Project-level virtual environment (optional)
├── data_dotquy/ # Folder containing PDF data
│ └── dotquy.pdf # Health document on stroke
│
├── src/ # Source code
│ ├── pycache/ # Python bytecode cache
│ ├── custom.py # Utility functions (e.g., for querying)
│ ├── load_split_data.py # Load and split PDF
│ └── vector_store.py # Vector store setup and retrieval
│
├── env/ # Environment configuration (optional)
├── cli.py # CLI interface for running the pipeline
├── dockerfile # Docker configuration
├── main.py # Main application entry point
├── README.md # Project documentation (this file)
├── requirements.txt # Required Python dependencies
└── run_qdrant.sh # Script to run Qdrant vector DB locally

---

## ⚙️ Setup Instructions

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # On Linux/macOS
.venv\Scripts\activate           # On Windows
2. Install dependencies
pip install -r requirements.txt
pip install pip install langchain_google_genai
pip install qdrant_client
pip install dotenv
pip install langchain_community
pip install pypdf


🚀 How to Use
Step 1: 🐳 Run Qdrant Locally
bash:
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v qdrant_data:/qdrant/storage qdrant/qdrant:latest

Step 2: collection data
bash:
python cli.py add data_dotquy --collection dotquy


Step 3: run main file to ask a question 
bash:
python main.py





📄 License
This project is released under the MIT License.

🧩 Notes
Ensure your PDF documents are high quality and relevant to your domain.

You can integrate any LLM provider (e.g., OpenAI, Cohere, or local models).

Supports swapping vector backends like FAISS, Chroma, or Qdrant with ease.
