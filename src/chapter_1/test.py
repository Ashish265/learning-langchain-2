from langchain_openai.chat_models import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model_name="gpt-3.5-turbo")
respone = model.invoke("The sky is")
print(respone.content)
