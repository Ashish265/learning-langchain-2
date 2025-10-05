from langchain_text_splitters import RecursiveCharacterTextSplitter

PYTHON_CODE = """
def hello_world():
    print("Hello, world!")

# CALL THE FUNCTION
hello_world()
"""

python_splitter = RecursiveCharacterTextSplitter.from_language(
    chunk_size=50,
    chunk_overlap=10,
    language="python"
)

python_docs = python_splitter.create_documents(PYTHON_CODE)
print(python_docs)
