from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage

from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

model = ChatOllama(model="llama3.1")

# for chunk in model.stream([
#         HumanMessage(content="Hi! I'm Bob"),
#         AIMessage(content="Hello Bob! How can I assist you today?"),
#         HumanMessage(content="What's my name?"),
#     ]):
#     print(chunk,end="",flush=True)

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


# with_message_history = RunnableWithMessageHistory(model, get_session_history)

# config = {"configurable": {"session_id": "abc2"}}

# for chunk in with_message_history.stream(
#     [
#         HumanMessage(content="Hi! I'm Bob")
#     ],
#     config=config,
# ):
#     print(chunk.content,end="",flush=True)


# for chunk in with_message_history.stream(
#     [
#         HumanMessage(content="What's my name?")
#     ],
#     config=config
# ):
#     print(chunk.content,end="",flush=True)

# config = {"configurable": {"session_id": "abc3"}}

# for chunk in with_message_history.stream(
#     [
#         HumanMessage(content="What's my name?")
#     ],
#     config=config
# ):
#     print(chunk.content,end="",flush=True)
    
# config = {"configurable": {"session_id": "abc2"}}

# for chunk in with_message_history.stream(
#     [
#         HumanMessage(content="What's my name?")
#     ],
#     config=config
# ):
#     print(chunk.content,end="",flush=True)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = prompt | model

# for chunk in chain.stream(
#     {"messages": [HumanMessage(content="hi! I'm bob")], "language": "Spanish"},
# ):
#     print(chunk.content,end="",flush=True)

with_message_history = RunnableWithMessageHistory(chain, get_session_history)

config = {"configurable": {"session_id": "abc5"}}

response = with_message_history.invoke(
    [HumanMessage(content="Hi! I'm Jim")],
    config=config,
)

print(response.content)

response = with_message_history.invoke(
    [HumanMessage(content="What's my name?")],
    config=config,
)

print(response.content)

# for chunk in with_message_history.stream(
#     [HumanMessage(content="What's my name?")],
#     config=config,
# ):
#     print(chunk.content,end="",flush=True)
    
# with_message_history = RunnableWithMessageHistory(
#     chain,
#     get_session_history,
#     input_messages_key="messages",
# )

# config = {"configurable": {"session_id": "abc11"}}


# for chunk in chain.stream(
#     {"messages": [HumanMessage(content="hi! I'm bob")], "language": "Spanish"},
#     config=config,
# ):
#     print(chunk.content,end="",flush=True)

# for chunk in chain.stream(
#     {"messages": [HumanMessage(content="whats my name?")], "language": "Spanish"},
#     config=config,
# ):
#     print(chunk.content,end="",flush=True)