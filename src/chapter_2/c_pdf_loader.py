from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("/workspaces/learning-langchain-2/test.pdf")
docs = loader.load()
print(docs)
