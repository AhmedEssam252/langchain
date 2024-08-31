from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM


template = """Question: {question}

Answer: Let's think step by step."""

prompt = ChatPromptTemplate.from_template(template)

model = OllamaLLM(model="llama3.1")

chain = prompt | model

# print(chain.invoke({"question": "what is langchain?"}))

for chunk in chain.stream({"question": "what is langchain?"}):
    print(chunk,end="",flush=True)