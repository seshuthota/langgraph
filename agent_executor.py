import operator
import os
from keys import TAVILY_API_KEY, LS_API_KEY
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import DuckDuckGoSearchRun
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_functions_agent
from typing import TypedDict, Annotated, List, Union
from langchain_core.messages import BaseMessage
from langchain_core.agents import AgentAction, AgentFinish
from langgraph.prebuilt.tool_executor import ToolExecutor
from langgraph.graph import END, StateGraph

os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = LS_API_KEY

# tools = [TavilySearchResults(max_results=1)]

tools = [DuckDuckGoSearchRun()]
prompt = hub.pull("hwchase17/openai-functions-agent")
llm = ChatOpenAI(base_url="http://localhost:1234/v1", model="gpt-3.5-turbo-1106", streaming=True, api_key='not-needed')

agent_runnable = create_openai_functions_agent(llm, tools, prompt)


class AgentState(TypedDict):
    input: str
    chat_history: list[BaseMessage]
    agent_outcome: Union[AgentAction, AgentFinish, None]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]


tool_executor = ToolExecutor(tools)


# Define an agent
def run_agent(data):
    agent_outcome = agent_runnable.invoke(data)
    assert isinstance(agent_outcome, object)
    return {"agent_outcome": agent_outcome}


# Define function to execute tools
def execute_tools(data):
    agent_action = data["agent_outcome"]
    output = tool_executor.invoke(agent_action)
    return {"intermediate_steps": [(agent_action, str(output))]}


# Define logic that will decide which conditional edge to go down
def should_continue(data):
    if isinstance(data["agent_outcome"], AgentFinish):
        return "end"
    else:
        return "continue"


# Define new graph
workflow = StateGraph(AgentState)

# Define nodes to cycle between
workflow.add_node("agent", run_agent)
workflow.add_node("action", execute_tools)

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",
        "end": END
    }
)

workflow.add_edge("action", "agent")
workflow.set_entry_point("agent")
app = workflow.compile()

inputs = {"input": "When is the IPL 2023 going to start?", "chat_history": []}
for s in app.stream(inputs):
    print(list(s.values())[0])
    print("\n----\n")
