from dotenv import load_dotenv
import os

# take environment variables from .env
load_dotenv()

# OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Pinecone API key
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Pinecone Environment
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")

# Pinecone Index
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

# Document Directory Path
DOC_DIR_PATH = "/documents/"

# OpenAI LLM Model
OPENAI_LLM = "text-davinci-003"
