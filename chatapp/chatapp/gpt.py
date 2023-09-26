import os

import openai
from langchain import GoogleSearchAPIWrapper
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain.tools import Tool

from chatapp.store import query_db

BASE_PROMPT_TEMPLATE_PATH = os.path.dirname(os.path.abspath(__file__)) + "/../data/prompt/question.txt"
SEARCH_PROMPT_TEMPLATE_PATH = os.path.dirname(
    os.path.abspath(__file__)) + "/../data/prompt/determine_if_results_are_relevant.txt"


def read_prompt_template(path):
    with open(path, 'r') as f:
        return f.read()


openai.api_key = os.environ["OPENAI_API_KEY"]

llm = ChatOpenAI(model_name='gpt-3.5-turbo-16k', temperature=0.9)

chat_history = ChatMessageHistory()

chat_memory = ConversationBufferMemory(
    llm=llm,
    memory_key="chat_histories",
    input_key="user_message",
    chat_memory=chat_history,
)

qa_chain = LLMChain(
    llm=llm,
    prompt=ChatPromptTemplate.from_template(
        template=read_prompt_template(BASE_PROMPT_TEMPLATE_PATH)
    ),
    verbose=True,
    memory=chat_memory
)

# 7. 최적화를 위해 외부 application을 이용하여 구현해도 된다.(예: web search 기능)
search = GoogleSearchAPIWrapper(
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    google_cse_id=os.getenv("GOOGLE_CSE_ID"),
)

search_tool = Tool(
    name="Google Search",
    description="Search Google for recent results.",
    func=search.run,
)

SEARCH_PROMPT_TEMPLATE_PATH = os.path.dirname(
    os.path.abspath(__file__)) + "/../data/prompt/determine_if_results_are_relevant.txt"

search_value_check_chain = LLMChain(
    llm=llm,
    prompt=ChatPromptTemplate.from_template(
        template=read_prompt_template(SEARCH_PROMPT_TEMPLATE_PATH)
    ),
    verbose=True,
)


def ask_question(question: str) -> str:
    context = dict(question=question)

    related_web_search_results = search_tool.run(question)
    has_value = search_value_check_chain.run(context)

    if has_value == 'Y':
        context['search_results'] = related_web_search_results

    context['related_documents'] = query_db(question)
    context['chat_histories'] = chat_memory.buffer

    answer = qa_chain.run(question, context=context)

    chat_history.add_user_message(question)
    chat_history.add_ai_message(answer)

    return answer
