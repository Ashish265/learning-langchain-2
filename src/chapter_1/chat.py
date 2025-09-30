from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

prompt = [HumanMessage("what is the captial of France")]

response = model.invoke(prompt)
print(response.content)
