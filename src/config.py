from dotenv import load_dotenv
import os

load_dotenv()

# Embedding service parameters
PRIMARY_QUEERY_EMBEDDING_PROVIDER = "gemini" # "gemini" or "local"
PRIMARY_GEMINI_EMBEDDING_MODEL = "gemini-embedding-2"
GEMINI_API_KEY = os.getenv("gemini_api_key")

FALLBACK_QUERY_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

EMBEDDING_BATCH_SIZE = 32
EMBEDDING_SHOW_PROGRESS_BAR = True

# Groq LLM parameters (used with LangChain)
GROQ_MODEL = "openai/gpt-oss-20b"  # Fast, active Groq model with full daily token quota
GROQ_API_KEY = os.getenv("groq_api_key")
GROQ_TEMPERATURE = 0.1

# Chunking parameters (sliding window)
CHUNK_SIZE = 200          # target number of words per chunk
CHUNK_OVERLAP = 20        # number of words to overlap between chunks

# Retrieval parameters
RETRIEVAL_TOP_K = 3       # number of chunks to retrieve for a query
ANSWER_TOP_K = 10         # number of top chunks to use for answer generation

# PDF processing thresholds (used in parser)
PDF_MIN_TEXT_PER_PAGE = 10  # skip pages with fewer than this many characters (likely images/charts)
YEAR_HEADER_LINES_PER_PAGE = 22   # number of lines per page inspected for year detection
YEAR_HEADER_MAX_PAGES = 3         # maximum number of pages to inspect for year detection