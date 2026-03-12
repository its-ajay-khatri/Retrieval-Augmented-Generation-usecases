from langchain_community.vectorstores import FAISS
from loader import get_embeddings, load_and_split

def ingest_javascript():

    docs = load_and_split("docs/javascript.pdf")

    embeddings = get_embeddings()

    vectordb = FAISS.from_documents(
        docs,
        embedding=embeddings
    )

    vectordb.save_local("vectorstores/javascript_index")

    print("javascript docs indexed")


if __name__ == "__main__":
    ingest_javascript()