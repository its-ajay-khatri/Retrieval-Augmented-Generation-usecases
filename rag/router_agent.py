from rag.retriever import load_retriever
# from rag.chains import build_chain
from rag.chains import generate_answer

# python_chain = build_chain(load_retriever("vectorstores/python_index"))
# java_chain = build_chain(load_retriever("vectorstores/java_index"))
# mern_chain = build_chain(load_retriever("vectorstores/mern_index"))
# js_chain = build_chain(load_retriever("vectorstores/javascript_index"))



python_retriever = load_retriever("vectorstores/python_index")
js_retriever = load_retriever("vectorstores/javascript_index")
mern_retriever = load_retriever("vectorstores/mern_index")
java_retriever = load_retriever("vectorstores/java_index")



# all_chains = [
#     python_chain,
#     js_chain,
#     java_chain,
#     mern_chain
# ]


# def route_question(question):

#     q = question.lower()

#     if "python" in q:
#         return python_chain.run(question)

#     elif "javascript" in q:
#         return js_chain.run(question)
    
#     elif "java" in q:
#         return java_chain.run(question)

#     elif "mern" in q or "react" in q:
#         return mern_chain.run(question)

#     # 2️⃣ Fallback → search all RAG pipelines
#     answers = []

#     for chain in all_chains:

#         try:
#             result = chain.invoke(question)

#             if result and "I don't know" not in str(result):
#                 answers.append(result)

#         except Exception:
#             pass

#     # 3️⃣ Return best answer if found
#     if answers:
#         return answers[0]["result"]

#     # 4️⃣ Out of scope
#     return "Sorry, this question is outside the knowledge base."


    # else:
    #     return "I don't know which knowledge base to use."


#------------------------------OR----------------------------------


def route_question(question):

    docs = python_retriever.invoke(question)

    if docs:
        return generate_answer(question, docs)

    docs = mern_retriever.invoke(question)

    if docs:
        return generate_answer(question, docs)
    

    docs = js_retriever.invoke(question)

    if docs:
        return generate_answer(question, docs)
    
    docs = java_retriever.invoke(question)

    if docs:
        return generate_answer(question, docs)

    return "Sorry, this question is outside the knowledge base."