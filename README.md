# 📄 RAG PDF Question Answering System

A **Retrieval-Augmented Generation (RAG)** based application that allows users to upload PDF documents and ask questions. The system retrieves relevant information from the document and generates accurate, context-aware answers using LLMs.

---

## 🚀 Features

* 📂 Upload PDF documents
* ❓ Ask questions based on uploaded content
* 🔍 Semantic search using vector embeddings
* ⚡ Fast retrieval with FAISS vector database
* 🤖 Context-aware responses using LLM
* 🌐 REST API built with Flask

---

## 🧠 How It Works

1. Upload PDF document
2. Extract and split text into chunks
3. Convert text into embeddings
4. Store embeddings in FAISS vector database
5. Convert user query into embedding
6. Retrieve most relevant chunks
7. Generate final answer using LLM

---

## 🛠️ Tech Stack

* Python
* Flask
* LangChain
* FAISS
* Groq LLM (LLaMA model)
* Embeddings

---

## ⚙️ Installation & Setup

### 1. Clone Repository

```bash
git clone https://github.com/Chiragkava81/rag-pdf-qa-system.git
cd rag-pdf-qa-system
```

---

### 2. Create Virtual Environment

```bash
python -m venv venv
```

Activate:

**Windows:**

```bash
venv\Scripts\activate
```

**Mac/Linux:**

```bash
source venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Setup Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key_here
HF_TOKEN=your_huggingface_token
```

---

### 5. Run the Application

```bash
flask --app app run
```

---

## 📌 Example

* Upload a PDF
* Ask: *"What is this document about?"*
* Get accurate answer based on document context
  
