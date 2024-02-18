import json
import operator
import os
from datetime import datetime
from keys import TAVILY_API_KEY, LS_API_KEY
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import DuckDuckGoSearchRun
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import format_tool_to_openai_function, convert_pydantic_to_openai_function
from langchain.agents import create_openai_functions_agent
from typing import TypedDict, Annotated, List, Union, Sequence
from langchain_core.messages import BaseMessage, FunctionMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.agents import AgentAction, AgentFinish
from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation
from langgraph.graph import END, StateGraph
from langchain_core.pydantic_v1 import BaseModel, Field
import warnings

warnings.filterwarnings("ignore")

os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = LS_API_KEY

tools = [TavilySearchResults()]

tool_executor = ToolExecutor(tools)

llm = ChatOpenAI(base_url="http://localhost:1234/v1", model="Llama-2-7B", streaming=True, api_key='not-needed')

# llm = ChatOpenAI(model="gpt-3.5-turbo-1106", streaming=True)


# class Response(BaseModel):
#     """Final response to the user"""
#     temperature: float = Field(description="The temperature")
#     other_notes: str = Field(description="Any other notes about the weather")


functions = [format_tool_to_openai_function(t) for t in tools]
# functions.append(convert_pydantic_to_openai_function(Response))
model = llm.bind_functions(functions)


# Define agent state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]

    if "function_call" not in last_message.additional_kwargs:
        return "end"
    elif last_message.additional_kwargs["function_call"]["name"] == "Response":
        return "end"
    else:
        return "continue"


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
    response = tool_executor.invoke(action)
    function_message = FunctionMessage(content=str(response), name=action.tool)
    return {"messages": [function_message]}


def first_model(state):
    human_input = state["messages"][-1].content
    return {
        "messages": [
            AIMessage(
                content="You are a helpful assistant that searches the internet. ",
                additional_kwargs={
                    "function_call": {
                        "name": "tavily_search_results_json",
                        "arguments": json.dumps({"query": human_input})
                    }
                }
            )
        ]
    }


workflow = StateGraph(AgentState)
workflow.add_node("first_agent", first_model)
workflow.add_node("agent", call_model)
workflow.add_node("action", call_tool)

workflow.set_entry_point("first_agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",
        "end": END
    }
)

workflow.add_edge("action", "agent")
workflow.add_edge("first_agent", "action")

app = workflow.compile()

inputs = {"messages": [HumanMessage(content="What is the weather in NY?")]}
for output in app.stream(inputs):
    for key, value in output.items():
        print(f"Output from node : {key}")
        print("---")
        print(value)
    print("\n---\n")
