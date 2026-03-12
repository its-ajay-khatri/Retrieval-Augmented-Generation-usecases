from langchain_classic.chains import RetrievalQA
from config.llm_config import get_llm
from langchain_core.prompts import PromptTemplate


# def build_chain(retriever):

#     llm = get_llm()

#     chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         retriever=retriever,
#         chain_type="stuff"
#     )

#     return chain




#----------------------OR----------------------------------

llm = get_llm()

prompt = PromptTemplate.from_template("""
You are a helpful assistant.

When answering, follow these rules:
- Base your response ONLY on the context below.
- Use precise years, model names, and performance stats when available.
- Do NOT use or infer any external information."

Context:
{context}

Question:
{question}
""")


chain = prompt | llm


def generate_answer(query, retrieved_docs):

    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    response = chain.invoke({
        "context": context,
        "question": query
    })

    return response