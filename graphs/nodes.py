from typing import Annotated, Literal, Sequence, TypedDict

from langchain import hub
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate
from tools.retriever_tool import RetrieverTool
from utils.ingest_data import VectorDB
from prompts.rag_prompt import rag_prompt, agent_system_prompt
import pprint

### Edges
grass_cutter_vector_db = VectorDB(doc_store_path="./data/stihl")
grass_cutter_tool = RetrieverTool(grass_cutter_vector_db, "grass_cutter_retriever", "Search and retrieve information about grass cutter/power tiller. Use it to retrieve information like instructions of using it, spare parts, available accessories, new products available etc")
tools = [grass_cutter_tool.get_tool()]

def grade_documents(state) -> Literal["generate", "rewrite"]:
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (messages): The current state

    Returns:
        str: A decision for whether the documents are relevant or not
    """

    print("---CHECK RELEVANCE---")

    # Data model
    class grade(BaseModel):
        """Binary score for relevance check."""

        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    # LLM
    model = ChatOpenAI(temperature=0, model="gpt-4o", streaming=True)

    # LLM with tool and validation
    llm_with_tool = model.with_structured_output(grade)

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. Documents can be texts or tables.\n 
        Here is the retrieved document(s): \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )

    # Chain
    chain = prompt | llm_with_tool

    messages = state["messages"]
    last_message = messages[-1]

    question = state["query"]
    docs = last_message.content

    scored_result = chain.invoke({"question": question, "context": docs})

    score = scored_result.binary_score

    if score == "yes":
        print(f"---DECISION: DOCS RELEVANT with question: {question}---")
        return "generate"
    else:
        print(f"---DECISION: DOCS NOT RELEVANT, fetched docs: ---{docs}")
        print(score)
        return "rewrite"


### Nodes


def agent(state):
    """
    Invokes the agent model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply end.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with the agent response appended to messages
    """
    messages = state["messages"]
    model = ChatOpenAI(temperature=0, streaming=True, model="gpt-4o")
    model = model.bind_tools(tools)
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", agent_system_prompt.template),
            MessagesPlaceholder("chat_history"),
        ]
    )

    question = messages[-1].content
    print(f"--- CALL AGENT WITH question: {question} --- \n")
    agent_retriever = qa_prompt | model
    response = agent_retriever.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response], "query": question}


def rewrite(state):
    """
    Transform the query to produce a better question.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with re-phrased question
    """

    print("---TRANSFORM QUERY---")
    question = state["query"]

    msg = [
        HumanMessage(
            content=f""" \n 
    Look at the input and try to reason about the underlying semantic intent / meaning. \n 
    Here is the initial question:
    \n ------- \n
    {question} 
    \n ------- \n
    Formulate an improved question: """,
        )
    ]

    # Grader
    print(f"---BEFORE TRANSFORMED QUERY: {question}---")
    model = ChatOpenAI(temperature=0, model="gpt-4o", streaming=True)
    response = model.invoke(msg)
    print(f"---TRANSFORMED QUERY: {response.content}---")
    return {"messages": [response], "query": response.content}

def generate(state):
    """
    Generate answer

    Args:
        state (messages): The current state

    Returns:
         dict: The updated state with re-phrased question
    """
    print("---GENERATE---")
    messages = state["messages"]
    question = state['query']
    last_message = messages[-1]
    docs = last_message.content

    # Prompt
    prompt = rag_prompt

    # LLM
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0, streaming=True)

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # Run
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}