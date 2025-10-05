from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()

template = ChatPromptTemplate.from_messages([
    ('system', """ You are a helpful assistant"""),
    ('human', '{question}'),
])

model = ChatOpenAI()

chatbot = template | model

response = chatbot.invoke({"question": "What is the capital of France?"})
print(response.content)
