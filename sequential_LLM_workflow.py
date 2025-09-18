import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph,START,END
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel,Field
from typing import Optional,TypedDict



load_dotenv()

llm=ChatGroq(
    model="groq/compound-mini"
)


parser=StrOutputParser()

class article_blog(BaseModel):
    topic:str
    article:Optional[str]=None
    blog:Optional[str]=None

def article(state:article_blog):
    template1=PromptTemplate(
        template="Basd on the mentioned topic. Create an article about {topic}",
        input_variables=['topic']
    )
    chain1=template1 | llm | parser
    res1=chain1.invoke({'topic':state.topic})
    return {'article':res1}

def blog(state:article_blog):
    template1=PromptTemplate(
        template="Based on the mentioned article. Create a short blog not more than 1000 words based on: {article}",
        input_variables=['article']
    )
    chain1=template1 | llm | parser
    res2=chain1.invoke({'article':state.article})
    return {'blog':res2}
## define graph
graph=StateGraph(article_blog)

## add nodes
graph.add_node('article',article)
graph.add_node('blog',blog)

## add edges
graph.add_edge(START,'article')
graph.add_edge('article','blog')
graph.add_edge('blog',END)

## compile graph
workflow=graph.compile()

## initial state
initial_state={'topic':'Future of AI Agents in India'}

## result
output=workflow.invoke(initial_state)

print(output)