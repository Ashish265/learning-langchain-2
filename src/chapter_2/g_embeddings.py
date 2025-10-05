from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()


model = OpenAIEmbeddings(model="text-embedding-3-small")

embeddings = model.embed_documents([
    "Hello world!",
    "Oh hello there!",
    "what's your name?",
    "where do you live?",
    "how old are you?"
])
print(embeddings)
