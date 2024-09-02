from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, trim_messages

from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough

import operator


model = ChatOllama(model="llama3.1")

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# for chunk in model.stream([
#         HumanMessage(content="Hi! I'm Bob"),
#         AIMessage(content="Hello Bob! How can I assist you today?"),
#         HumanMessage(content="What's my name?"),
#     ]):
#     print(chunk,end="",flush=True)


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
            "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# chain = prompt | model

# for chunk in chain.stream(
#     {"messages": [HumanMessage(content="hi! I'm bob")], "language": "Spanish"},
# ):
#     print(chunk.content,end="",flush=True)

# with_message_history = RunnableWithMessageHistory(
#     chain,
#     get_session_history,
#     input_messages_key="messages",
#     )

# config = {"configurable": {"session_id": "abc11"}}


# for chunk in with_message_history.stream(
#     {"messages": [HumanMessage(content="hi! I'm ahmed")], "language": "Spanish"},
#     config=config,
# ):
#     print(chunk.content,end="",flush=True)

# for chunk in with_message_history.stream(
#     {"messages": [HumanMessage(content="whats my name?")], "language": "Spanish"},
#     config=config,
# ):
#     print(chunk.content,end="",flush=True)
    


messages = [
    SystemMessage(content="you're a good assistant"),
    HumanMessage(content="hi! I'm bob"),
    AIMessage(content="hi!"),
    HumanMessage(content="I like vanilla ice cream"),
    AIMessage(content="nice"),
    HumanMessage(content="whats 2 + 2"),
    AIMessage(content="4"),
    HumanMessage(content="thanks"),
    AIMessage(content="no problem!"),
    HumanMessage(content="having fun?"),
    AIMessage(content="yes!"),
]

trimmer = trim_messages(
    max_tokens=65,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

try:
    # Try to invoke the messages first
    trimmer.invoke(messages)
except Exception as e:
    print(f"Error during message trimming: {e}")

try:
    chain = (
        RunnablePassthrough.assign(
            messages=operator.itemgetter("messages") | trimmer,
        )
        | prompt
        | model
    )

    response = chain.invoke(
        {
            "messages": messages + [HumanMessage(content="what's my name?")],
            "language": "English",
        }
    )
        
    print(response.content)
except Exception as e:
    # Handle any exceptions during the execution
    print(f"Error during chain invocation: {e}")

