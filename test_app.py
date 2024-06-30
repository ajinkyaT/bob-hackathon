from langchain.schema import AIMessage, HumanMessage
from graphs.rag_graph import agentic_rag_graph
import pprint

def sample_function():
    user_input = input("Enter some text: ")
    inputs = {"messages": [HumanMessage(content=user_input)]}
    for event in agentic_rag_graph.stream(inputs,stream_mode="values"):
            if "messages" in event:
                 event["messages"][-1].pretty_print()

sample_function()
