from langchain_community.vectorstores import FAISS
from loader import get_embeddings, load_and_split

def ingest_mern():

    docs = load_and_split("docs/mern.pdf")

    embeddings = get_embeddings()

    vectordb = FAISS.from_documents(
        docs,
        embedding=embeddings
    )

    vectordb.save_local("vectorstores/mern_index")

    print("MERN docs indexed")


if __name__ == "__main__":
    ingest_mern()