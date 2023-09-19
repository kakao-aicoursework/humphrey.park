# state.py
import asyncio
import os

import openai
import reflex as rx
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

# 참조: https://python.langchain.com/docs/use_cases/chatbots
from langchain.memory import ConversationSummaryMemory

# 참조: https://python.langchain.com/docs/modules/model_io/prompts/prompt_templates/
from langchain.prompts import (
    PromptTemplate
)

openai.api_key = os.environ["OPENAI_API_KEY"]

llm = ChatOpenAI(model_name='gpt-3.5-turbo-16k', temperature=0.9)

def loadFile(file_path: str) -> str:
    with open(file_path, "r") as f:
        str = f.read()
    return str


memory = ConversationSummaryMemory(llm=llm)
# 카카오싱크에 대한 데이터를 초기 대화 기록에 남겨둠
memory.save_context({"input": loadFile('./assets/project_data_카카오싱크.txt')}, {"output": ''})

prompt = PromptTemplate(
    input_variables=["q"],
    template="{q}"
)

chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory
)

class State(rx.State):
    # def handle_key_press(self, event):
    #     if event.type_ == 'key_down' and event.key == 'Enter':
    #         self.answer()

    # The current question being asked.
    question: str

    # Keep track of the chat history as a list of (question, answer) tuples.
    chat_history: list[tuple[str, str]]

    async def answer(self):
        # Our chatbot is not very smart right now...
        answer = "I don't know!"
        self.chat_history.append((self.question, ""))

        # Clear the question input.
        self.question = ""
        # Yield here to clear the frontend input before continuing.
        yield

        answer = chain.run(self.question)
        memory.save_context({"input": self.chat_history[-1][0]}, {"output": answer})

        for i in range(len(answer)):
            # Pause to show the streaming effect.
            await asyncio.sleep(0.1)
            # Add one letter at a time to the output.
            self.chat_history[-1] = (
                self.chat_history[-1][0],
                answer[: i + 1],
            )
            yield