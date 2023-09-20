# ChatApp
## 목적
프로젝트 2단계 LangChain Library 적용해보기

## 구성
- https://reflex.dev/docs/tutorial/final-app/ 템플릿을 기반으로 했고
- `chatapp/state.py` 파일에서 LLMChain을 이용했음

## 컨셉

### Step1. text 파일을 ConversationSummaryMemory로 저장해두고
```python
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

# 참조: https://python.langchain.com/docs/use_cases/chatbots
from langchain.memory import ConversationSummaryMemory

# 참조: https://python.langchain.com/docs/modules/model_io/prompts/prompt_templates/
from langchain.prompts import (
    PromptTemplate
)


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
```

### Step 2. chain을 이용해 API 호출

```python
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

    answer = chain.run(self.chat_history[-1][0])

    print(f'답변: {answer}')

    self.chat_history[-1] = (
        self.chat_history[-1][0],
        answer.replace('\n', '<br />'),
    )

    yield

    print(f'처리 완료')
    # Toggle the processing flag.
    self.processing = False
```