from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryByteStore
import uuid


load_dotenv()

prefix = 'postgresql+psycopg'
sufix = 'langchain:langchain@host.docker.internal:6024/langchain'

# See docker command above to launch a postgres instance with pgvector enabled.
connection = "://".join([prefix, sufix])
collection_name = "summaries"
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

# load the document
loader = TextLoader(
    "/workspaces/learning-langchain-2/test.txt",
    encoding="utf8",
)
docs = loader.load()

print("length of loaded docs:",
      len(docs[0].page_content))

# split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
chunks = text_splitter.split_documents(docs)

# Prompt
prompt_text = "summarize the following documnets: \n\n{doc}"

prompt = ChatPromptTemplate.from_template(prompt_text)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

summarize_chain = {
    "doc": lambda x: x.page_content
    } | prompt | llm | StrOutputParser()

# batch the chain across chunks
summaries = summarize_chain.batch(chunks,
                                  {'max_concurrency': 5})


# The vectorstore to use to index the child chunks
vectorstore = PGVector(
    embeddings=embeddings_model,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,
)

# The storage layer for the parent documents
store = InMemoryByteStore()
id_key = "doc_id"

# indexing the summaries in our vector store,
#  whilst retaining the original
# documents in our document store:
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    byte_store=store,
    id_key=id_key,
)

# Changed from summaries to chunks since we need same length as docs
doc_ids = [str(uuid.uuid4()) for _ in chunks]

# Each summary is linked to the original document by the doc_id
summary_docs = [
    Document(page_content=s, metadata={id_key: doc_ids[i]})
    for i, s in enumerate(summaries)
]

# Add the document summaries
# to the vector store for similarity search
retriever.vectorstore.add_documents(summary_docs)

# Store the original documents in the document store,
#  linked to their summaries via doc_ids
# This allows us to first search summaries efficiently,
#  then fetch the full docs when needed
retriever.docstore.mset(list(zip(doc_ids, chunks)))

# vector store retrieves the summaries
sub_docs = retriever.vectorstore.similarity_search(
    "chapter on philosophy", k=2)

print("sub docs: ", sub_docs[0].page_content)

print("length of sub docs:\n", len(sub_docs[0].page_content))

# Whereas the retriever will return the larger source document chunks:
retrieved_docs = retriever.invoke("chapter on philosophy")

print("length of retrieved docs: ", len(retrieved_docs[0].page_content))
