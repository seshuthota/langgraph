import os
import functools
import json
import operator
from typing import Annotated, TypedDict, Sequence

from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_community.output_parsers.ernie_functions import JsonOutputFunctionsParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.python import PythonREPL
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage
)

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation
from langchain_core.utils.function_calling import format_tool_to_openai_function
from keys import LS_API_KEY, TAVILY_API_KEY, GOOGLE_API_KEY

os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = LS_API_KEY

repl = PythonREPL()
tavily_tool = TavilySearchResults(max_results=5)

members = ["Researcher", "Coder"]

llm = ChatOpenAI(model="gpt-3.5-turbo-16k-0613", streaming=True)


# code_llm = ChatOpenAI(base_url="http://localhost:1234/v1", model="deepseek-coder", api_key="not-needed")
# llm = ChatGoogleGenerativeAI(model="gemini-pro", api_key=GOOGLE_API_KEY)


def create_agent(llm_input: ChatOpenAI, tools: list, system_prompt_input: str):
    # Each worker node will be given a name and some tools.
    agent_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt_input,
            ),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm_input, tools, agent_prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor


def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}


@tool
def python_repl(
        code: Annotated[str, "The python code to execute to generate your chart."]
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    return f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"


system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    " following workers:  {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)
# our team supervisor is an LLM node. It just picks the next agent to process
# and decides when the work is completed.
options = ["FINISH"] + members

function_def = {
    "name": "route",
    "description": "Select the next role.",
    "parameters": {
        "title": "routeSchema",
        "type": "object",
        "properties": {
            "next": {
                "title": "Next",
                "anyOf": [
                    {"enum": options}
                ],
            }
        },
        "required": ["next"],
    },
}

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            system_prompt,
        ),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Given the conversation above, who should act next?"
            "Or should we FINISH? select one of :  {options}",
        ),
    ]
).partial(options=str(options), members=", ".join(members))

supervisor_chain = (
        prompt
        | llm.bind_functions(functions=[function_def], function_call="route")
        | JsonOutputFunctionsParser()
)


# The agent state is the input to each node in the graph
class AgentState(TypedDict):
    # The annotation tells the graph that new messages will always
    # be added  to the current state
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # The 'next' field indicates where to route to next
    next: str


research_agent = create_agent(llm, [tavily_tool],
                              "You are a web researcher.")

research_node = functools.partial(agent_node, agent=research_agent, name="Researcher")

code_agent = create_agent(llm, [python_repl], "You may generate safe python code to analyze data and generate "
                                              "charts using matplotlib.")
code_node = functools.partial(agent_node, agent=code_agent, name="Coder")

# Define workflow graph

workflow = StateGraph(AgentState)
workflow.add_node("supervisor", supervisor_chain)
workflow.add_node("Researcher", research_node)
workflow.add_node("Coder", code_node)

for member in members:
    # Workers to always report back to supervisor
    workflow.add_edge(member, "supervisor")
# Supervisor populates the"next" field in the graph state
# which routes to a node or finishes
conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)

# Finally, we set the entry point to the supervisor
workflow.set_entry_point("supervisor")

graph = workflow.compile()

for s in graph.stream({
    "messages": [
        HumanMessage(content="Code a sine wave plot and save the result to disk.")
    ],
}):
    if "__end__" not in s:
        print(s)
        print("\n-----\n")
