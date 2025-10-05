from langchain_community.document_loaders import TextLoader

loader = TextLoader("/workspaces/learning-langchain-2/test.txt",
                    encoding="utf8")
docs = loader.load()
print(docs)
