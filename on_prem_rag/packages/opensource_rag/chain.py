import os
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import VLLMOpenAI
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from langchain_community.document_loaders import WikipediaLoader
from langchain_qdrant import Qdrant, QdrantVectorStore
from operator import itemgetter
from langchain_core.runnables.passthrough import RunnablePassthrough
from typing import Any
from pydantic import BaseModel
from ollama import Client

# caching
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
if not os.path.exists("langchain_cache"): os.mkdir("langchain_cache")
set_llm_cache(SQLiteCache(database_path="langchain_cache/sqlite_cache.db"))


# constants
vllm_docker_address = "http://vllm:8000/v1"   # Not "http://localhost:8888/v1" because the container is running in the same network by docker-compose
vllm_model = "deepseek-ai/deepseek-llm-7b-chat"
ollama_docker_address = "http://ollama:11434"   # Not "http://localhost:8888/v1" because the container is running in the same network by docker-compose
embedding_model = 'mxbai-embed-large'
qdrant_docker_address = 'http://qdrant:6333'  # Not "http://localhost:6333" because the container is running in the same network by docker-compose
wikipedia_page_name = '2024 NBA playoffs'
collection_name = 'NBAplayoffs2024'


# prompt
rag_prompt_template = '''\
<｜begin▁of▁sentence｜>You are a helpful assistant. You answer User Query based on provided Context. If you can't answer the User Query with the provided Context, say you don't know.

User: 

Question: {question}

Context: {context}

Assistant: \
'''
rag_prompt = PromptTemplate.from_template(rag_prompt_template).with_config({'run_name':'rag_prompt'})



# llm
llm = VLLMOpenAI(
    openai_api_key="fake_api_key",  
    openai_api_base=vllm_docker_address,  
    model_name=vllm_model
).with_config({'run_name':'vllm_deepseek-llm-7b-chat'})




# embedding model
client = Client(host=ollama_docker_address)
client.pull(embedding_model)
embeddings = OllamaEmbeddings(model=embedding_model, base_url=ollama_docker_address)


# vector store & retriever
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
client = QdrantClient(qdrant_docker_address)

if collection_name not in list([x.name for x in client.get_collections().collections]):
    loader = WikipediaLoader(wikipedia_page_name)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    vectorstore = Qdrant.from_documents(
        chunks,
        embeddings,
        location=qdrant_docker_address, 
        collection_name=collection_name
    )


else:
    vectorstore = QdrantVectorStore(
        client = client,
        collection_name=collection_name,
        embedding=embeddings,
        )
retriever = vectorstore.as_retriever(search_kwargs  = {'k':1}).with_config({'run_name':'retriever_mxbai-embed-large'})


# Input-Output types for FastAPI
class Input(BaseModel):
    question: str

class Output(BaseModel):
    answer: str


# RAG chain
rag_chain = (
        {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
        | RunnablePassthrough.assign(context= lambda x : '\n\n'.join([ y.metadata['summary'] for y in x["context"] ])    )
        | rag_prompt | llm
    ).with_config({'run_name':'RAG_chain'}).with_types(input_type=Input, output_type=Output)

