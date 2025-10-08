from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain
from dotenv import load_dotenv

load_dotenv()

prefix = 'postgresql+psycopg'
sufix = 'langchain:langchain@host.docker.internal:6024/langchain'

# See docker command above to launch a postgres instance with pgvector enabled.
connection = "://".join([prefix, sufix])

raw_documents = TextLoader(
    "/workspaces/learning-langchain-2/test.txt",
    encoding="utf8",
).load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
documents = text_splitter.split_documents(raw_documents)

embeddings_model = OpenAIEmbeddings()

db = PGVector.from_documents(
    documents,
    embeddings_model,
    connection=connection
)


retriver = db.as_retriever(search_kwargs={"k": 2})

query = 'Who are the key figures in the ancient greek history of philosophy?'

docs = retriver.invoke(query)
# print(docs[0].page_content)

prompt = ChatPromptTemplate.from_template(
    """Answer the question based on the following
    context: {context} Question: {Question}.""")

llm = ChatOpenAI()

llm_chain = prompt | llm

result = llm_chain.invoke(
    {"context": docs, "Question": query}
)

# print(result)
# print("\n\n")

print("Running again but this time encapsulate the logic for efficiency\n")


@chain
def qa_chain(query: str) -> str:
    docs = retriver.invoke(query)
    formatted = prompt.invoke(
        {"context": docs, "Question": query}
    )
    return llm.invoke(formatted)


result = qa_chain.invoke(query)
print(result)
