from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain
from dotenv import load_dotenv
load_dotenv()

template = ChatPromptTemplate.from_messages([
    ('system', """ You are a helpful assistant"""),
    ('human', '{question}'),
])

model = ChatOpenAI()


@chain
def chatbot(values):
    prompt = template.invoke(values)
    return model.invoke(prompt)


response = chatbot.invoke({"question": "What is the capital of France?"})
print(response.content)
