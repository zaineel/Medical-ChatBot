from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.vectorstores import faiss
from langchain_community.vectorstores import FAISS


DATA_PATH = "data"
DB_FAISS_PATH = "vectorstores/faiss_db"

# Create a vector database
def create_vector_db(data):
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    documents =  loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
    chunks = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2",
    model_kwargs = {
        'device': 'cpu',
    })
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(DB_FAISS_PATH)


if __name__ == "__main__":
    create_vector_db(DATA_PATH)