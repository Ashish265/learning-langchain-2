from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prefix = 'postgresql+psycopg'
sufix = 'langchain:langchain@host.docker.internal:6024/langchain'

connection = "://".join([prefix, sufix])

raw_document = TextLoader(
    "/workspaces/learning-langchain-2/test.txt", encoding="utf8",
).load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
documents = text_splitter.split_documents(raw_document)
embeddings_model = OpenAIEmbeddings()
db = PGVector.from_documents(
    documents,
    embeddings_model,
    connection=connection
)
retriever = db.as_retriever(search_kwargs={"k": 5})

prompt_hyde = ChatPromptTemplate.from_template(
    """ Please write a passage to answer the question below.
    Question: {question} \n Passage:"""
)

generated_doc = prompt_hyde | ChatOpenAI(temperature=0) | StrOutputParser()


print(generated_doc.invoke({
    """question": "Who are some lesser known
      philosophers in the ancient greek history of philosophy?"""}))

retrieval_chain = generated_doc | retriever

query = """Who are some lesser known
 philosophers in the ancient greek history of philosophy?"""

prompt = ChatPromptTemplate.from_template(
    """Answer the question based on the following
    context: {context} Question: {question}.""")
llm = ChatOpenAI(temperature=0)


@chain
def qa(input):
    docs = retrieval_chain.invoke(input)

    formatted = prompt.invoke({"context": docs, "question": input})

    answer = llm.invoke(formatted)
    return answer


print(""" Running hyde \n""")
result = qa.invoke(query)
print("\n\n")
print(result.content)
