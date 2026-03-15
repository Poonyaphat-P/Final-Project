import os
# test
from dotenv import load_dotenv # Use .env to hide API key
load_dotenv()  # This loads the variables from the .env file

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from tavily import TavilyClient # Make your chatbot can search information on the internet
from langchain.tools import tool
from typing import Dict, Any # For the tavily part

from langgraph.checkpoint.memory import MemorySaver #For the memory
from langchain.agent import create_agent 

#from langchain_openai import OpenAI
#from langchain.chains import APIChain

system_prompt = "You are a friendly chatbot that talk like a kids."

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
memory = MemorySaver()
config = {"configurable": {"thread_id": "1"}}



#Initilize the websearch tool
@tool
def web_search(query: str) -> Dict[str, Any]:

    """Search the web for information"""

    return tavily_client.search(query)

tools = [web_search]

llm = ChatOpenAI(
    model="gpt-4o-mini",
    max_tokens=100
)

scifi_agent = create_react_agent(
    model=llm,
    checkpointer=memory,
    tools=tools,
    prompt=system_prompt
)

while True:
    user_input = input("You: ")
    if user_input == "exit":
        break
    response = scifi_agent.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config
    )
    print(response['messages'][-1].content)
