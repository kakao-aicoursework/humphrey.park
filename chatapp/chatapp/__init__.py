import os
import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

openai.api_key = os.environ["OPENAI_API_KEY"]

from langchain.document_loaders import (
    TextLoader,
)

CHROMA_COLLECTION_NAME = "kakao-bot"
CHROMA_PERSIST_DIR = os.path.dirname(os.path.abspath(__file__)) + "/../data/chroma-persist"

# 1. 사용할 데이터(project_data_카카오소셜.txt, project_data_카카오싱크.txt, project_data_카카오톡채널.txt)를 preprocessing을 하여 사용한다.
PROJECT_DATA_CATALOG = ["카카오소셜", "카카오싱크", "카카오톡채널"]
DATA_DIR = os.path.dirname(os.path.abspath(__file__)) + "/../data/project"
PROJECT_DATA_PATHS = [f"{DATA_DIR}/project_data_{catalog}.txt" for catalog in PROJECT_DATA_CATALOG]

print(PROJECT_DATA_PATHS)

LOADER_DICT = {
    "txt": TextLoader
}

# 2. PROJECT_DATA_PATHS 데이터를 chromadb의 client를 통해 VectorDB에 업로드하여 사용한다.
for project_data_path in PROJECT_DATA_PATHS:
    loader = LOADER_DICT.get(project_data_path.split(".")[-1])
    if loader is None:
        raise ValueError("Not supported file type")
    documents = loader(project_data_path).load()

    text_splitter = CharacterTextSplitter(chunk_size=510, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    Chroma.from_documents(
        docs,
        OpenAIEmbeddings(),
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name=CHROMA_COLLECTION_NAME,
        collection_metadata={"hnsw:space": "cosine"})
