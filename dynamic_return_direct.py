import json
import operator
import os
from datetime import datetime
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
from langchain_core.pydantic_v1 import BaseModel, Field

os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = LS_API_KEY


class SearchTool(BaseModel):
    """Look up things online, optionally return directly"""
    query: str
    return_direct: bool = Field(
        default=False,
        description="Whether or the result of this should be returned directly to the user without seeing what it is")


tools = [DuckDuckGoSearchRun(args_schema=SearchTool)]

tool_executor = ToolExecutor(tools)

# llm = ChatOpenAI(base_url="http://localhost:1234/v1", model="gpt-3.5-turbo-1106", streaming=True, api_key='not-needed')
llm = ChatOpenAI(model="gpt-3.5-turbo-1106", streaming=True)

functions = [format_tool_to_openai_function(t) for t in tools]

model = llm.bind_functions(functions)


# Define agent state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


# Define should continue
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    if "function_call" not in last_message.additional_kwargs:
        return "end"
    else:
        arguments = json.loads(last_message.additional_kwargs["function_call"]["arguments"])
        if arguments.get("return_direct", False):
            return "final"
        else:
            return "continue"


# Define the function that calls the model
def call_model(state):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


def call_tool(state):
    messages = state["messages"]
    last_message = messages[-1]
    tool_name = last_message.additional_kwargs["function_call"]["name"]
    arguments = json.loads(last_message.additional_kwargs["function_call"]["arguments"])
    if tool_name == "DuckDuckGoSearchRun":
        if "return_direct" in arguments:
            del arguments["return_direct"]
    action = ToolInvocation(
        tool=tool_name,
        tool_input=arguments
    )
    response = tool_executor.invoke(action)
    function_message = FunctionMessage(content=str(response), name=action.tool)
    return {"messages": [function_message]}


workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("action", call_tool)
workflow.add_node("final", call_tool)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",
        "final": "final",
        "end": END
    }
)

workflow.add_edge("action", "agent")
workflow.add_edge("final", END)

app = workflow.compile()

inputs = {"messages": [HumanMessage(content="What is the meaning of life? return answer directly to the user by "
                                            "setting return_direct = True")]}

for output in app.stream(inputs):
    for key, value in output.items():
        print(f"Output from node '{key}' : ")
        print("---")
        print(value)
    print("\n---\n")
