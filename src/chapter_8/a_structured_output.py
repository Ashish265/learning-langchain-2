from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


class Joke(BaseModel):

    setup: str = Field(..., description="The setup of Joke")
    punchline: str = Field(..., description="The puchline of Joke")


model = ChatOpenAI(model="gpt-4o")
model = model.with_structured_output(Joke)

result = model.invoke("Tell me a joke about cats")
print(result)
