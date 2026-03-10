import os
from dotenv import load_dotenv

from llama_index.core import load_index_from_storage
from llama_index.core import StorageContext
from llama_index.core import Settings
from llama_index.llms.gemini import Gemini


# 1 Load environment variables
load_dotenv()

# Get API key from .env (recommended)
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")

# 2. Configure Gemini LLM
Settings.llm = Gemini(
    model="models/gemini-2.5-flash",
    api_key=api_key,
    context_window=1000000, # Set to 1 million for Gemini 2.5
    num_output=2048,        # Reserve 2k tokens for the answer
)

# Also explicitly update the Settings to be safe
Settings.context_window = 1000000
Settings.num_output = 2048        #num output limit
 
print("📂 Loading stored Tree Index...")
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)

query_engine = index.as_query_engine(
     response_mode="tree_summarize"
)

print("\n💬 Ask your questions (type 'exit' to quit)\n")

while True:
    question = input("You: ")

    if question.lower() == "exit":
        print("👋 Exiting...")
        break

    response = query_engine.query(question)

    print("\n🤖 Answer:\n")
    print(response)
    print("\n" + "-" * 60 + "\n")