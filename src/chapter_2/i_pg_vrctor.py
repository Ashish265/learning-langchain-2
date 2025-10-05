from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_core.documents import Document
from dotenv import load_dotenv
import uuid

load_dotenv()

prefix = 'postgresql+psycopg'
sufix = 'langchain:langchain@localhost:6024/langchain'

# See docker command above to launch a postgres instance with pgvector enabled.
connection = "://".join([prefix, sufix])

# load the document, split it into chunks
raw_docs = TextLoader(
    "/workspaces/learning-langchain-2/test.txt",
    encoding="utf8",
).load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
documents = text_splitter.split_documents(raw_docs)

# create embeddings
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

db = PGVector.from_documents(
    documents,
    embeddings_model,
    connection=connection
)

results = db.similarity_search("query", k=4)

print(results)

print("Adding documents to the vector store")
ids = [str(uuid.uuid4()) for _ in range(len(documents))]
db.add_documents(
    [
        Document(
            page_content="there are cats in the pond",
            metadata={"location": "pond", "topic": "animals"},
        ),
        Document(
            page_content="ducks are also in the pond",
            metadata={"location": "pond", "topic": "animals"},
        ),
    ],
    ids=ids,
)

print("Documents added successfully. \n Fetched documents:",
      len(db.get_by_ids(ids)))

print("Deleting documents with ids:", ids[1])
db.delete({"ids": ids})

print("Documents deleted successfully. \n Fetched documents:",
      len(db.get_by_ids(ids)))
