from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langserve import RemoteRunnable

parser = StrOutputParser()

model = OllamaLLM(model="llama3.1")


# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()

# template = """sentence: {sentence}

# Answer: Translate only the following from English into Italian """

# prompt = ChatPromptTemplate.from_template(template)

# chain = messages | model

# for chunk in chain.stream({"sentence": "write a poem about python"}):
#     print(chunk,end="",flush=True)

# chain = prompt | model

#  print(chain.invoke({"question": "what is langchain?"}))


# messages = [
#     SystemMessage(content="Translate the following from English into Italian"),
#     HumanMessage(content="Hi"),
# ]

# result = model.invoke(messages)
# parser.invoke(result)
# chain = model | parser

system_template = "Translate the following into {language} directly don't explain:"

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

chain = prompt_template | model | parser


# for chunk in chain.stream({"language": "arabic", "text": "i need to translte this word 'write a poem about python'"}):
#     print(chunk,end="",flush=True)


remote_chain = RemoteRunnable("http://localhost:8000/chain/")
remote_chain.invoke({"language": "italian", "text": "hi"})
#write a code to add output in a file instead of print it
