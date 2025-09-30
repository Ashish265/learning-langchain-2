from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

system_msg = SystemMessage(
    """You are a helpful assistant that responds
    to questions with three exclamation marks."""
    )
human_msg = HumanMessage("What is the capital of France")

response = model.invoke([system_msg, human_msg])
print(response.content)
