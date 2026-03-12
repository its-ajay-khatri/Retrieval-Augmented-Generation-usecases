from langchain_ollama import OllamaLLM

def get_llm():

    llm = OllamaLLM(
        model="llama3.2:3b",
        temperature=0
    )

    return llm