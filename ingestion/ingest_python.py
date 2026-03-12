from langchain_community.vectorstores import FAISS
from loader import get_embeddings, load_and_split

def ingest_python():

    docs = load_and_split("docs/python.pdf")

    embeddings = get_embeddings()

    vectordb = FAISS.from_documents(
        docs,
        embedding=embeddings
    )

    vectordb.save_local("vectorstores/python_index")

    print("Python docs indexed")


if __name__ == "__main__":
    ingest_python()