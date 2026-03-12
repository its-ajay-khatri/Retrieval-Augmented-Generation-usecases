from langchain_community.vectorstores import FAISS
from ingestion.loader import get_embeddings

def load_retriever(path):

    embeddings = get_embeddings()

    vectordb = FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectordb.as_retriever(search_kwargs={"k":3})