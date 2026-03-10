# rag_example.py ✅ (LangChain 1.0.3 Compatible, New API)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableMap

# === 1️⃣ Load and Split Text ===
with open("sample.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_text(raw_text)
print(f"✅ Loaded {len(chunks)} text chunks")

# === 2️⃣ Create Embeddings ===
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# === 3️⃣ Create Chroma Vector Store ===
vector_db = Chroma.from_texts(
    texts=chunks,
    embedding=embedding_model,
    persist_directory="./chroma_store"
)
vector_db.persist()
retriever = vector_db.as_retriever(search_kwargs={"k": 4})

# === 4️⃣ Initialize Local LLM (Ollama) ===
llm = Ollama(model="llama3.2:3b")

# === 5️⃣ Define Prompt Template ===
prompt_template = """
You are a professional automotive historian who has studied every Lamborghini ever produced.

When answering, follow these rules:
- Base your response ONLY on the context below.
- Use precise years, model names, and performance stats when available.
- Do NOT use or infer any external information.

Context:
{context}

Question:
{question}

Answer:
"""
PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# === 6️⃣ Create RAG Chain using LCEL ===
def format_docs(docs):
    """Combine document texts into one context string."""
    return "\n\n".join([d.page_content for d in docs])

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | PROMPT
    | llm
)

# === 7️⃣ Ask a Question ===
query = input("\nAsk something about your text: ")
result = rag_chain.invoke(query)

print("\n--- Answer ---")
print(result)
