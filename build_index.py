import os
from dotenv import load_dotenv

from llama_index.core import SimpleDirectoryReader
from llama_index.core import SummaryIndex
from llama_index.core import Settings
from llama_index.llms.gemini import Gemini


# 1 Load environment variables
load_dotenv()

# Get API key from .env (recommended)
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")

# gemini-2.5-flash is fast and cost-effective for indexing/summarization
# 2. Configure Gemini LLM
Settings.llm = Gemini(
    model="models/gemini-2.5-flash",
    api_key=api_key,
    context_window=1000000, # Set to 1 million for Gemini 2.5
    num_output=2048,        # Reserve 2k tokens for the answer
)

# Also explicitly update the Settings to be safe
Settings.context_window = 1000000
Settings.num_output = 2048

print("📄 Loading PDF...")
documents = SimpleDirectoryReader("./data").load_data()

print("🌳 Building Summary Tree Index (Page Indexing)...")
index = SummaryIndex.from_documents(
    documents,
    show_progress=True
)

print("💾 Persisting index to disk...")
index.storage_context.persist("./storage")

print("✅ Index built and saved successfully!")