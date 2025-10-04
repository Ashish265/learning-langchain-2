from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

template = ChatPromptTemplate.from_messages([
    ('system', """
    Answer the question based on the context below.
     If the question cannot be answered
     based on the context,say "I don't know"."""),
    ('human', 'Context:{context}'),
    ('human', 'Question:{question}'),
])

model = ChatOpenAI()

prompt = template.invoke(
    {
        "context": """
        The most recent advancements in NLP
        are being driven by Large Language Models (LLMs).
        These models outperform their smaller counterpart and
        have become invaluable for developers
        who are creating applications with NLP capabilities.
        Developers can tap into these models
        through Hugging Face's `transformers` library,
        or by utilizing OpenAI and Cohere's offerings
        through the `openai` and `cohere` libraries, respectively.
        """,
        "question": "Which model providers offer LLMs?",
    }
)

response = model.invoke(prompt)

print(response.content)
