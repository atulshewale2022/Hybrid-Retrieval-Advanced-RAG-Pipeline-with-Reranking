# Hybrid-Retrieval Advanced RAG Pipeline with Reranking

A high-performance Retrieval-Augmented Generation (RAG) system using an advanced **Hybrid Search Architecture** (Dense Semantic Vector + Sparse BM25 Keyword Search) coupled with a **CrossEncoder Reranker** for maximal precision.

## 🚀 Key Engineering Features
* **Hybrid Search Strategy:** Pairs dense vector similarities via ChromaDB with exact keyword lookups via BM25 to catch edge case contexts.
* **CrossEncoder Re-ranking:** Maximises response accuracy by scoring retrieved nodes using `ms-marco-MiniLM-L-6-v2` before feeding context to the LLM.
* **O(N) De-duplication:** Custom-built array filtering pipelines prevent overlapping context bloat, speeding up inference metrics.
* **Deterministic Output Execution:** Handled via Groq's high-speed inference engine using Llama 3.3 70B with a zero-temperature parameter setting.

## 🛠️ Technical Stack
* **Framework:** LangChain
* **LLM Engine:** Groq API (`llama-3.3-70b-versatile`)
* **Vector Store:** ChromaDB
* **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)
* **Reranker:** SentenceTransformers CrossEncoder

## 📦 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/atulshewale2022/Hybrid-Retrieval-Advanced-RAG-Pipeline-with-Reranking.git
   cd Hybrid-Retrieval-Advanced-RAG-Pipeline-with-Reranking
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables:**
   Create a `.env` file in the root directory and add your Groq API key:
   ```env
   GROQ_API_KEY=your_actual_groq_api_key
   ```

4. **Run the RAG Chatbot:**
   ```bash
   python rag_with_hyb_search_reranking.py
   ```
