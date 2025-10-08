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

# Instruction to generate multiple queries

prespective_prompt = ChatPromptTemplate.from_template(
    """You are an AI language model assistant.
      Your task is to generate five different versions
        of the given user question to retrieve
          relevant documents from a vector database.
    By generating multiple perspectives on the user question,
      your goal is to help the user overcome some of the
        limitations of the distance-based  similarity search.
    Provide these alternative questions separated by newlines.
    Original question: {question} """
)

llm = ChatOpenAI()


def parse_queries_output(message):
    return message.content.split("\n")


query_gen = prespective_prompt | llm | parse_queries_output


def get_unique_union(document_lists):
    deduped_docs = {
        doc.page_content: doc for sublist in document_lists for doc in sublist
    }
    return list(deduped_docs.values())


retrieval_chain = query_gen | retriever.batch | get_unique_union

prompt = ChatPromptTemplate.from_template(
    """Answer the question based on the following
    context: {context} Question: {question}.""")

query = "Who are the key figures in the ancient greek history of philosophy?"


@chain
def multi_query_qa(input):
    # fetch relevant documents

    docs = retrieval_chain.invoke(input)  # format prompt
    formatted = prompt.invoke(
        {"context": docs, "question": input})  # generate answer
    answer = llm.invoke(formatted)
    return answer


# run
print("Running multi query qa\n")
result = multi_query_qa.invoke(query)
print(result.content)
