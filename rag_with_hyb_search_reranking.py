import os
from typing import List
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma 
from sentence_transformers import CrossEncoder 

pdf_path = r'C:\Users\Atul Shewale\Python_Practice\Gen AI\offer_letter.pdf'
documents = PyPDFLoader(pdf_path).load()
print(f"Total pages loaded: {len(documents)}")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, 
    chunk_overlap=100, 
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = text_splitter.split_documents(documents)
print(f"Total structured chunks generated: {len(chunks)}")

embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

vectorstore = Chroma.from_documents(documents=chunks, embedding=embedding_model)

chroma_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 3

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

prompt = ChatPromptTemplate.from_template(
"""
You are an expert AI assistant.

Answer ONLY using the supplied context.

If the answer is not present in the context,
reply exactly:

"I couldn't answer as i dont have information."

Context:
{context}

Question:
{question}
"""
)

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
chain = prompt | llm

print('*' * 50)
print("********** Chatbot Active **********")
print('*' * 50, '\n')

while True:
    user_query = input('User: ')

    if user_query.strip().lower() in ['exit', 'quit']:
        print("Chatbot session ended. Goodbye!")
        break

    if not user_query.strip():
        continue

    chroma_results = chroma_retriever.invoke(user_query)
    bm25_results = bm25_retriever.invoke(user_query)

    seen_texts = set()
    hybrid_results = []
    for doc in (chroma_results + bm25_results):
        if doc.page_content not in seen_texts:
            seen_texts.add(doc.page_content)
            hybrid_results.append(doc)

    pairs = [(user_query, doc.page_content) for doc in hybrid_results]
    scores = reranker.predict(pairs)
    
    reranked_results = sorted(zip(hybrid_results, scores), key=lambda x: x[1], reverse=True)
    top_documents = [doc for doc, score in reranked_results[:3]]

    unique_docs = []
    unique_text = set()
    for doc in top_documents:
        if doc.page_content not in unique_text:
            unique_text.add(doc.page_content)
            unique_docs.append(doc)

    context_blocks = []
    for i, doc in enumerate(unique_docs, start=1):
        page_num = doc.metadata.get('page', 'N/A')
        block = f"\n______________________________________________\n doc : {i} \n page no : {page_num} \n content : {doc.page_content}"
        context_blocks.append(block)
    
    final_context = "".join(context_blocks)

    final_result = chain.invoke({'context': final_context, 'question': user_query}).content

    print(f'Answer: {final_result}\n')
