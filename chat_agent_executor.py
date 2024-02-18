import operator
import os
from keys import TAVILY_API_KEY, LS_API_KEY
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import DuckDuckGoSearchRun
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import format_tool_to_openai_function
from langchain.agents import create_openai_functions_agent
from typing import TypedDict, Annotated, List, Union, Sequence
from langchain_core.messages import BaseMessage, FunctionMessage, HumanMessage, SystemMessage
from langchain_core.agents import AgentAction, AgentFinish
from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation
from langgraph.graph import END, StateGraph

os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = LS_API_KEY

tools = [DuckDuckGoSearchRun()]

tool_executor = ToolExecutor(tools)

# prompt = hub.pull("hwchase17/openai-functions-agent")
llm = ChatOpenAI(base_url="http://localhost:1234/v1", model="gpt-3.5-turbo-1106", streaming=True, api_key='not-needed')

functions = [format_tool_to_openai_function(t) for t in tools]

model = llm.bind_functions(functions)


# Define Agent State
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


# Define the function that deteermines whether to continue or not
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    if "function_call" not in last_message.additional_kwargs:
        return "end"
    else:
        return "continue"


# Define the function that calls the mode
def call_model(state):
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}


def call_tool(state):
    messages = state["messages"]
    last_message = messages[-1]

    action = ToolInvocation(
        tool=last_message.additional_kwargs["function_call"]["name"],
        tool_input=last_message.additional_kwargs["function_call"]["arguments"]
    )
    tool_name = last_message.additional_kwargs["function_call"]["name"],
    tool_input_data = last_message.additional_kwargs["function_call"]["arguments"]
    print(f"In Tool Invocation with {tool_name} as tool and with input as {tool_input_data}")
    response = tool_executor.invoke(action)
    function_message = FunctionMessage(content=str(response), name=action.tool)
    return {"messages": [function_message]}


workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("action", call_tool)
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,

    {
        "continue": "action",
        "end": END
    }
)
workflow.add_edge("action", "agent")
app = workflow.compile()

inputs = {"messages": [SystemMessage(
    content="You are a helpful assistant."
            "You always fulfill the task given with what you have"
            " You can access tool to look up online for realtime information when needed"),
    HumanMessage(content="What is the weather in Mumbai Today?")]}
print(app.invoke(inputs))
