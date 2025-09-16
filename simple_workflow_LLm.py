from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph,START,END
from typing import TypedDict

# os.getenv("GROQ_API_KEY")
load_dotenv()

llm=ChatGroq(
    model="llama-3.1-8b-instant"
)

## define class for Query
class LLM_query(TypedDict):
    question:str
    answer:str

## define Ask_llm
def Ask_llm(state:LLM_query):
    question=state['question']
    result=llm.invoke(question)
    state['answer']=result.content
    return state

## define the Graph
graph=StateGraph(LLM_query)

## add nodes in graph
graph.add_node('Query_node',Ask_llm)

## add edges in graph
graph.add_edge(START,'Query_node')
graph.add_edge('Query_node',END)

## compile the graph
workflow=graph.compile()

##execute the graph
initial_state={'question':'How many planets in solar system'}

final_state=workflow.invoke(initial_state)

print(final_state['answer'])

# result=llm.invoke("Who is the president of America")
# print(result.content)