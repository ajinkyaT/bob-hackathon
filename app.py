from langchain_groq import ChatGroq
from langchain.schema import AIMessage, HumanMessage
import gradio as gr
from graphs.rag_graph import agentic_rag_graph
import pprint


llm = ChatGroq(temperature=0, model_name="llama3-8b-8192")

def predict(message, history):
    history_langchain_format = []
    print(f"History: {pprint.pformat(history)}")
    for human, ai in history[1:]:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    history_langchain_format.append(HumanMessage(content=message))
    inputs = {"messages": history_langchain_format}
    return agentic_rag_graph.invoke(inputs)['messages'][-1].content
    # partial_message = ""
    # for output in agentic_rag_graph.stream(inputs, stream_mode="updates"):
    #     # if "messages" in output:
    #     #          partial_message += pprint.pformat(output["messages"][-1]) + "\n\n"
    #     #          yield partial_message
    #     for key, value in output.items():
    #         partial_message += f"Output from key: {key}" + "\n"
    #         partial_message += pprint.pformat(pprint.pformat(value)) + "\n\n"
    #         yield  partial_message

# gr.ChatInterface(predict).queue().launch()
chatbot = gr.Chatbot([[None,"Hello Ishwar! Thank you for purchasing the STIHL FS120 Brush Cutter. I am Suhani, your customer assistant. If you need any help regarding the FS120 as well as STIHL products, feel free to ask me."]])
gr.ChatInterface(predict,chatbot=chatbot, examples=["Hi I need to service my grass cutter where can I service it?", "Please suggest me suitable accesory for my grass cutter."], title="After Sales Service").launch()