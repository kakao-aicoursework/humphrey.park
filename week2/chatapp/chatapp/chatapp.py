# chatapp.py

from chatapp.state import State

from rxconfig import config

import reflex as rx
from chatapp import style

filename = f"{config.app_name}/{config.app_name}.py"


def qa(question: str, answer: str) -> rx.Component:
    return rx.box(
        rx.box(
            rx.text(question, style=style.question_style),
            text_align="right",
        ),
        rx.box(
            rx.html(answer, style=style.answer_style),
            text_align="left",
        ),
        margin_y="1em",
    )


def chat() -> rx.Component:
    return rx.box(
        rx.foreach(
            State.chat_history,
            lambda messages: qa(messages[0], messages[1]),
        )
    )


def action_bar() -> rx.Component:
    return rx.form(
        rx.form_control(
            rx.hstack(
                rx.input(
                    placeholder="Ask a question",
                    id="question",
                    value=State.question,
                    on_change=State.set_question,
                    style=style.input_style,
                ),
                rx.button(
                    rx.text("Ask"),
                    type_="submit",
                    style=style.button_style,
                    is_disabled=State.processing,
                ),
            ),
            is_disabled=State.processing,
        ),
        on_submit=[State.answer, rx.set_value("question", "")],
        width="100%",
    )

def index() -> rx.Component:
    return rx.container(
        chat(),
        action_bar(),
    )


# Add state and page to the app.
app = rx.App()
app.add_page(index)
app.compile()
