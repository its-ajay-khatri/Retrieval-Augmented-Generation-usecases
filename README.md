This is a Retrieval-Augmented Generation (RAG) application that reads a PDF, stores its content as embeddings in a vector database, retrieves relevant chunks when a user asks a question, and generates an answer using a local LLM.

used FAISS(Facebook AI search for vector stores), a sample pdf reader,

initially install the requirements, run command "pip install -r requirements.txt"

Delete the faiss_index folder as it will be recreated using main.py file(kept faiss_index folder for reference)

then run command "py main.py"

python version used: 3.12
langchain version: 1.0.8
LLM: ollama's llama 3.2:3b

