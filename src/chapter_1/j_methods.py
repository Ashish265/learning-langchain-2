from langchain_openai.chat_models import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

model = ChatOpenAI(model="gpt-3.5-turbo")

completion = model.invoke("Hi there!")
print(completion.content)


completions = model.batch(["Hi there!", "Bye!"])
print(completions)

for token in model.stream("Bye!"):
    print(token)
    # Good
    # bye
    # !
