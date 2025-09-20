

from langchain_groq import ChatGroq

from langchain_core.output_parsers import StrOutputParser
from typing import TypedDict,Annotated,List,Optional,Literal
from pydantic import BaseModel,Field
from dotenv import load_dotenv
import os
import operator

from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage,AIMessage

load_dotenv()

llm1=ChatGroq(model="llama-3.1-8b-instant")

llm2=ChatGroq(model='llama-3.3-70b-versatile')

llm3=ChatGroq(model='llama-3.1-8b-instant')

class post_schema(BaseModel):
  topic:Annotated[str,Field(description="Topic for X post")]
  post:Annotated[str,Field(description="post about the topic")]

class evaluate_schema(BaseModel):
  post:str
  evaluation:Literal['approved','rejected']
  feedback:str

class optimize_schema(BaseModel):
  post:str
  max_iteration:int

class post_state(TypedDict):
  topic:str
  post:str
  evaluation:Literal['approved','rejected']
  feedback:str
  max_iteration:int
  feedback_history:Annotated[List[str],operator.add]
  tweet_history:Annotated[List[str],operator.add]

structure_llm1=llm1.with_structured_output(post_schema)
structure_llm2=llm2.with_structured_output(evaluate_schema)
structure_llm3=llm3.with_structured_output(optimize_schema)

def post_create(state:post_state):
  prompt=f"""You are an expert tutor who explains topics clearly and engagingly.
  Generate a well-structured post about the following topic: "{state['topic']}".

  Guidelines:
  - Keep the length under 1000 characters.
  - Write in simple, clear language (assume the audience is high school or early college level).
  - Include a brief example or analogy if useful.
  - Avoid jargon unless necessary, and define it if you use it."""

  result1=structure_llm1.invoke(prompt)
  return {'post':result1.post,'tweet_history':[result1.post]}

def evluate_post(state:post_state):
  prompt=f"""You are a strict reviewer whose ONLY job is to check if a generated post meets quality standards.
Here is the post:
"{state['post']}"

You must evaluate the post on ALL of the following criteria:

1. **Clarity and readability**: The language must be simple, clear, and audience-friendly. If the post uses confusing, irrelevant, or overly complex phrasing → REJECT.
2. **Length**: The post must be under 1000 characters. If it exceeds → REJECT.
3. **Relevance**: The post must stay directly on-topic: "{state['topic']}". If it drifts into unrelated or nonsense content → REJECT.
4. **Accuracy and quality**: The post must be informative, factually correct, and engaging. If it contains hallucinations, jokes, misleading info, or nonsense → REJECT.

Your response MUST be one of the following (and nothing else):
- `APPROVED` (only if ALL four criteria are satisfied).
- `REJECTED:  if it fails any of the criteria and  Give detailed reasons as a feedback for improvement.
  """
  result=structure_llm2.invoke(prompt)

  return {'evaluation':result.evaluation,'feedback':result.feedback,'feedback_history':[result.feedback]}

def optimize_post(state:post_state):
  prompt=f"""You are an editor improving a rejected post.
  Here is the original post:
  "{state['post']}"

  The evaluator rejected it for this reason: "{state['feedback']}".

  Revise the post to fix the issues while keeping the meaning intact.
  Guidelines:
  - Keep the post under 1000 characters.
  - Ensure clarity, engagement, and relevance to the topic "{state['topic']}".
  - Keep the writing style tutor-like and student-friendly.

  """
  iteration=state['max_iteration']
  result3=structure_llm3.invoke(prompt)
  return {'post':result3.post,'max_iteration':iteration+1,'tweet_history':[result3.post]}

def check_post(state:post_state):
  if state['evaluation']=='approved':
    return 'approved'
  else:
    return 'need improvement'

# structure_llm1.invoke(prompt.format(topic='AI in India',max_characters=500))

## Define graph
graph=StateGraph(post_state)

## add node
graph.add_node('post_create',post_create)
graph.add_node('evluate_post',evluate_post)
graph.add_node('optimize_post',optimize_post)

## add edge
graph.add_edge(START,'post_create')
graph.add_edge('post_create','evluate_post')
graph.add_conditional_edges('evluate_post',check_post,{'approved':END,'need improvement':'optimize_post'})
graph.add_edge('optimize_post','evluate_post')


workflow=graph.compile()

# from langchain_core.runnables.graph_mermaid import draw_mermaid_png

# from IPython.display import Image
# Image(workflow.get_graph().draw_mermaid_png())

initial_state={'topic':HumanMessage('corruption rise in India'),'max_iteration':1}

result=workflow.invoke(initial_state)

print(result)



