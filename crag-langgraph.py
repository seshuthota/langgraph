import os
from pprint import pprint
from typing import Dict, TypedDict

from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.schema import Document
from keys import TAVILY_API_KEY, LS_API_KEY
from langgraph.graph import StateGraph, END

os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = LS_API_KEY
os.environ["LANGCHAIN_PROJECT"] = "crag-langgraph"
embedding = GPT4AllEmbeddings()
local_llm = "mistral:instruct"

rag_prompt = hub.pull("rlm/rag-prompt")

URL = "https://lilianweng.github.io/posts/2023-06-23-agent/"

loader = WebBaseLoader(URL)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=100
)
all_splits = text_splitter.split_documents(docs)

vector_store = Chroma.from_documents(
    documents=all_splits,
    collection_name="crag-langgraph",
    embedding=embedding,
    persist_directory="chromadb",
)

db = Chroma(
    persist_directory="chromadb",
    collection_name="crag-langgraph",
    embedding_function=embedding
)

retriever = db.as_retriever()


# State of the graph
class GraphState(TypedDict):
    """
    Represents the state of the graph
    Attributes:
        keys: A dictionary where each key is a string and each value is any
    """
    keys: Dict[str, any]


# Nodes
def retrieve(state):
    """
    Retrieves a node from the graph
    Args:
        state (dict): The current state of the graph
    Returns:
        state (dict): New key added to the state, documents that contains the relevant documents
    """
    print("-----RETRIEVE-----")
    state_dict = state["keys"]
    question = state_dict["question"]

    documents = retriever.get_relevant_documents(question)
    return {"keys": {"documents": documents, "question": question}}


def generate(state):
    """
    Generates a node from the graph
    Args:
        state (dict): The current state of the graph
    Returns:
        state (dict): New key added to the state, response that contains the generated response
    """
    print("-----GENERATE-----")
    state_dict = state["keys"]
    documents = state_dict["documents"]
    question = state_dict["question"]

    llm = ChatOllama(model=local_llm, temperature=0)

    def format_docs(docs):
        return "\n".join(doc.page_content for doc in docs)

    rag_chain = rag_prompt | llm | StrOutputParser()

    generation = rag_chain.invoke({"context": format_docs(documents), "question": question})
    return {
        "keys": {
            "documents": documents,
            "question": question,
            "generation": generation
        }
    }


def grade_documents(state):
    """
    Determines whether retrieved documents are relevant to the question
    Args:
        state (dict): The current state of the graph
    Returns:
        state (dict): New key added to the state, grade that contains the grade of the documents
    """

    print("-----GRADE DOCUMENTS-----")
    state_dict = state["keys"]
    documents = state_dict["documents"]
    question = state_dict["question"]

    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
            Here is the retrieved document: \n\n {context} \n\n
            Here is the user question: {question} \n
            If the document contains keywords related to the user question, grade it as relevant. \n
            It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
            Provide the binary score as a JSON with a single key 'score' and no premable or explaination.""",
        input_variables=["question", "context"],
    )
    llm = ChatOllama(model=local_llm, format="json", temperature=0)

    chain = prompt | llm | JsonOutputParser()

    # Score
    filtered_docs = []
    search = "No"

    for d in documents:
        score = chain.invoke(
            {
                "question": question,
                "context": d.page_content
            }
        )
        grade = score["score"]
        if grade == "yes":
            print("---GRADE: DOCUMENT IS RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT IS NOT RELEVANT---")
            search = "Yes"
    return {
        "keys": {
            "documents": filtered_docs,
            "question": question,
            "run_web_search": search,
        }
    }


def transform_query(state):
    """
        Transform the query to produce a better question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates question key with a re-phrased question
    """

    print("----TRANSFORM QUERY-----")
    state_dict = state["keys"]
    question = state_dict["question"]

    prompt = PromptTemplate(
        template="""You are a language model that is trying to rephrase a user question to improve the quality of the search results. \n
        Here is the original question: {question} \n
        Please rephrase the question to make it more clear and concise. \n
        Provide an improved question without any preamble, only respond with the updated question: """,
        input_variables=["question"],
    )

    # Grader
    llm = ChatOllama(model=local_llm, temperature=0)

    chain = prompt | llm | StrOutputParser()

    better_question = chain.invoke(
        {
            "question": question
        }
    )

    return {
        "keys": {
            "documents": state_dict["documents"],
            "question": better_question,
        }
    }


def web_search(state):
    """
      Web search based on the re-phrased question using Tavily API.

      Args:
          state (dict): The current graph state

      Returns:
          state (dict): Web results appended to documents.
    """

    print("----WEB SEARCH-----")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]

    tool = TavilySearchResults()
    docs = tool.invoke({"query": question})
    web_results = "\n".join(d["content"] for d in docs)
    web_results = Document(page_content=web_results)
    documents.append(web_results)

    return {
        "keys": {
            "documents": documents,
            "question": question
        }
    }


def decide_to_generate(state):
    """
    Determines whether to generate an answer or re-generate a question for web search.

    Args:
        state (dict): The current state of the agent, including all keys.

    Returns:
        str: Next node to call
    """

    print("----DECIDE TO GENERATE-----")
    state_dict = state["keys"]
    search = state_dict["run_web_search"]

    if search == "Yes":
        print("----DECISION: TRANSFORM QUERY AND RUN WEB SEARCH----")
        return "transform_query"
    else:
        print("----DECISION: GENERATE RESPONSE----")
        return "generate"


# Build the graph

workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)
workflow.add_node("web_search", web_search)

workflow.set_entry_point("retrieve")

workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "generate": "generate",
        "transform_query": "transform_query"
    }
)

workflow.add_edge("transform_query", "web_search")
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()

inputs = {
    "keys": {
        "question": "What is the best way to learn a new language?"
    }
}

for output in app.stream(inputs):
    for k, v in output.items():
        print(f"Node '{k}'")
        print("-----OUTPUT-----")
        # print(v)
    print("\n-----END-----\n")

print(v['keys']['generation'])
