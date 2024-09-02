# client.py  
from langserve import RemoteRunnable  

remote_chain = RemoteRunnable("http://localhost:8000/chain/")  
# response = remote_chain.invoke()  


for chunk in remote_chain.stream({"language": "italian", "text": "hi"}):
    print(chunk,end="",flush=True)

# print(response)  