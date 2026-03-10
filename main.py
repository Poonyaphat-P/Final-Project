import os
import requests # For api requests
from dotenv import load_dotenv # Use .env to hide API key
load_dotenv()  # This loads the variables from the .env file

os.environ["OPENAI_API_KEY"] = os.getenv("OPENROUTER_API_KEY") #Make the program use the openrouter API instead of OpenAI

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, AIMessageChunk
from tavily import TavilyClient # Make your chatbot can search information on the internet
from langchain.tools import tool
from typing import Dict, Any # For the tavily part
from langchain.agents import create_agent

system_prompt = "You are a friendly chatbot that talks like a kid. Only use the web search tool when the user explicitly asks for current information or news. For casual conversation, just reply directly."

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

#Initilize the websearch tool
@tool
def web_search(query: str) -> Dict[str, Any]:

    """Search the web for information"""

    return tavily_client.search(query, max_results=3)

@tool
def get_airquality(city: str) -> Dict[str, Any]:
    """Get the air quality (AQI) of a specific city. Extract the city name from the user's prompt."""
    url = f"https://api.waqi.info/feed/{city}/?token={os.getenv('AQICN_API_KEY')}"
    response = requests.get(url)
    return response.json()

tools = [web_search, get_airquality]

llm = ChatOpenAI(
    model="gpt-4o-mini",
    max_tokens=70
)

chat_history = [SystemMessage(content=system_prompt)]
MAX_HISTORY = 8

scifi_agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt
)


while True:
    user_input = input("You: ")
    if user_input == "exit":
        break
    
    #Add user message
    chat_history.append(HumanMessage(content=user_input))

    #Limit memory size
    if len(chat_history) > MAX_HISTORY:
        chat_history = [chat_history[0]] + chat_history[-MAX_HISTORY:]

    # Stream the response token by token
    print("Bot: ", end="", flush=True)
    full_response = ""
    for token, metadata in scifi_agent.stream(
        {"messages": chat_history},
        stream_mode="messages"
    ):
        if isinstance(token, AIMessageChunk) and token.content:  # Use isinstance to prevent bot from printing json output from using tools
            print(token.content, end="", flush=True)
            full_response += token.content
    print()  # Move to a new line when done

    # Save the full response to memory
    chat_history.append(AIMessage(content=full_response))