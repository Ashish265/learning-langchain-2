from typing import Literal
from langchain_core.prompts import (
    ChatPromptTemplate, SystemMessagePromptTemplate,
    HumanMessagePromptTemplate)
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class RouteQuery(BaseModel):
    """ Route a userquery to most relevant datasource  """

    datasource: Literal["python_docs", "js_docs"] = Field(
        ...,
        description="The datasource to use to answer the user query",
    )


# prompt template
# LLM with functional call

llm = ChatOpenAI(model="gpt-4o", temperature=0)

structured_llm = llm.with_structured_output(RouteQuery)

system = """You are an expert at routing a user question
 to the appropriate data source.
 Based on the programming language
 the question is referring to,
 route it to the relevant data source."""

system_message = SystemMessagePromptTemplate.from_template(system)
human_message = HumanMessagePromptTemplate.from_template("{question}")

prompt = ChatPromptTemplate.from_messages([
    system_message,
    human_message
])

router = prompt | structured_llm

question = """Why doesn't the following code work:
from langchain_core.prompts
import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages(["human", "speak in {language}"])
prompt.invoke("french") """

result = router.invoke({"question": question})
print("\nRouting to: ", result)


def choose_route(result):
    if "python_docs" in result.datasource.lower():
        return "chain for python_docs"
    else:
        return "chain for js_docs"


full_chain = router | RunnableLambda(choose_route)

result = full_chain.invoke({"question": question})
print("\nChoose route: ", result)
