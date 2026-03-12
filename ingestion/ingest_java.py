from langchain_community.vectorstores import FAISS
from loader import get_embeddings, load_and_split

def ingest_java():

    docs = load_and_split("docs/java.pdf")

    embeddings = get_embeddings()

    vectordb = FAISS.from_documents(
        docs,
        embedding=embeddings
    )

    vectordb.save_local("vectorstores/java_index")

    print("Java docs indexed")

    
if __name__ == "__main__":
    ingest_java()