from langchain_openai.chat_models import ChatOpenAI
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()


class AnswerWithJustification(BaseModel):
    '''An answer to the user's question along with justification for the
    answer'''
    answer: str
    justification: str


llm = ChatOpenAI()
structured_llm = llm.with_structured_output(AnswerWithJustification)

structured_response = structured_llm.invoke(
    """what weighs more, a pound of feathers or a pound of bricks?""")

print(structured_response)
