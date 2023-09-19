# ChatApp
## 목적
프로젝트 2단계 LangChain Library 적용해보기

## 구성
- https://reflex.dev/docs/tutorial/final-app/ 템플릿을 기반으로 했고
- `chatapp/state.py` 파일에서 LLMChain을 이용했음

## 컨셉

### Step1. text 파일을 ConversationSummaryMemory로 저장해두고
```python
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
    # verbose=True,
    memory=memory
)
```

### Step 2. PromptTemplate을 이용해 API 호출

```python
 async def answer(self):
    # Add to the answer as the chatbot responds.
    answer = ""
    self.chat_history.append((self.question, answer))

    answer = chain.run(self.question)
    answer = answer.replace("\n", "<br />")
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

    # Clear the question input.
    self.question = ""

    # Yield here to clear the frontend input before continuing.
    yield
```