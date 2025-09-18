import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph,START,END
from typing import TypedDict

class calculater_state(TypedDict):
  a:int
  b:int
  add_r:int
  mul_r:float
  div_r:float
  result:str

def calculate_sum(state:calculater_state)->calculater_state:
  add=state['a']+state['b']
  return {'add_r':add}

def calculate_mul(state:calculater_state)->calculater_state:
  mul=state['a']*state['b']
  return {'mul_r':mul}

def calculate_div(state:calculater_state)->calculater_state:
  div=state['a']/state['b']
  return {'div_r':div}

def aggregation_result(state:calculater_state)->calculater_state:
  result=f"""The sum of number is {state['add_r']},
  The multiplication of number is {state['mul_r']},
  The division of number is {state['div_r']}"""
  return {'result':result}

## define graph
graph=StateGraph(calculater_state)

## add node
graph.add_node('calculate_sum',calculate_sum)
graph.add_node('calculate_mul',calculate_mul)
graph.add_node('calculate_div',calculate_div)
graph.add_node('aggregation_result',aggregation_result)

## add edges
graph.add_edge(START,'calculate_sum')
graph.add_edge(START,'calculate_mul')
graph.add_edge(START,'calculate_div')
graph.add_edge('calculate_sum','aggregation_result')
graph.add_edge('calculate_mul','aggregation_result')
graph.add_edge('calculate_div','aggregation_result')
graph.add_edge('aggregation_result',END)

## comile
workflow=graph.compile()

initial_state={'a':20,'b':6}

output=workflow.invoke(initial_state)

print(output)