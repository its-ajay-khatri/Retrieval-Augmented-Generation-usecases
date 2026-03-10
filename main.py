import os
import re
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.prompts import PromptTemplate


# ===============================
# 1️⃣ LOAD PDF
# ===============================
def load_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents


# ===============================
# 2️⃣ CLEAN / NORMALIZE
# ===============================
def clean_documents(docs):
    seen = set()
    cleaned_docs = []

    for doc in docs:
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", doc.page_content).strip()

        # Deduplicate exact pages
        if text not in seen:
            seen.add(text)
            doc.page_content = text
            cleaned_docs.append(doc)

    return cleaned_docs


# ===============================
# 3️⃣ CHUNKING
# ===============================
def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    return splitter.split_documents(docs)


# ===============================
# 4️⃣ CREATE VECTOR STORE
# ===============================
def create_vector_store(chunks):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    vectorstore.save_local("faiss_index")
    return vectorstore


# ===============================
# 5️⃣ LOAD EXISTING VECTOR STORE
# ===============================
def load_vector_store():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    return FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )


# ===============================
# 6️⃣ RETRIEVE DOCUMENTS (KNN/ANN)
# ===============================
def retrieve_docs(vectorstore, query, k=4):
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    return retriever.invoke(query)


# ===============================
# 7️⃣ GENERATE ANSWER WITH LLAMA3.2
# ===============================
def generate_answer(query, retrieved_docs):
    llm = OllamaLLM(model="llama3.2:3b")

    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    prompt = PromptTemplate.from_template("""
You are a helpful assistant.
                                              
When answering, follow these rules:
- Base your response ONLY on the context below.
- Use precise years, model names, and performance stats when available.
- Do NOT use or infer any external information.

Context:
{context}

Question:
{question}
""")

    chain = prompt | llm

    response = chain.invoke({
        "context": context,
        "question": query
    })

    return response


# ===============================
# 🚀 MAIN RAG PIPELINE
# ===============================
def rag_pipeline(pdf_path, query):
    # If index exists → load
    if Path("faiss_index").exists():
        print("Loading existing vector store...")
        vectorstore = load_vector_store()
    else:
        print("Creating new vector store...")
        docs = load_pdf(pdf_path)
        docs = clean_documents(docs)
        chunks = chunk_documents(docs)
        vectorstore = create_vector_store(chunks)

    retrieved_docs = retrieve_docs(vectorstore, query)
    answer = generate_answer(query, retrieved_docs)

    return answer


# ===============================
# 🏁 ENTRY POINT
# ===============================
if __name__ == "__main__":

    pdf_file = "nodejs_extra.pdf"

    print("\n📄 RAG Application Ready (Ollama + Llama3.2)")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("Ask a question: ")

        if query.lower() == "exit":
            break

        response = rag_pipeline(pdf_file, query)
        print("\n🤖 Answer:\n", response)
        print("\n" + "=" * 50 + "\n")