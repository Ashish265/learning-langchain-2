from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

loader = TextLoader(
    "/workspaces/learning-langchain-2/test.txt",
    encoding="utf8",
)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

splitted_docs = splitter.split_documents(docs)

print(splitted_docs)
