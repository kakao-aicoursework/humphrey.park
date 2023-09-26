# state.py
import reflex as rx

from chatapp.gpt import ask_question

class State(rx.State):
    # The current question being asked.
    question: str

    # Whether we are processing the question.
    processing: bool = False

    # Keep track of the chat history as a list of (question, answer) tuples.
    chat_history: list[tuple[str, str]]

    def answer(self):
        if len(self.question) < 1:
            return None

        # Set the processing flag to true and yield.
        self.processing = True

        print(f'질문: {self.question}')
        # img from https://icons8.com/preloaders/
        self.chat_history.append((self.question, "<img src='/loading.png'>"))

        # Clear the question input.
        self.question = ""

        yield

        answer = ask_question(self.chat_history[-1][0])

        print(f'답변: {answer}')

        self.chat_history[-1] = (
            self.chat_history[-1][0],
            answer.replace('\n', '<br />'),
        )

        yield

        print(f'처리 완료')
        # Toggle the processing flag.
        self.processing = False
