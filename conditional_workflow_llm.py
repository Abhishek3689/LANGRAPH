

import os
from dotenv import load_dotenv

from langgraph.graph import StateGraph,START,END
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel,Field
from typing import TypedDict,Annotated,Literal,List,Optional

load_dotenv()

llm=ChatGroq(model='llama-3.3-70b-versatile')

class review_sentiment(BaseModel):
  sentiment:Annotated[str,Field(description='sentiment of the review')]

class Diagnosis_schema(BaseModel):
  tone:Literal['anger','frustrated','disappointed','calm']=Field(description='emotional tone expressed by the customer')
  urgency:Literal['Low','Medium','High']=Field(description='urgency of the customer')
  specificity:Literal['bug','delay','performance','quality','other']=Field(description="are the customer is talking about any specific issue")

structure_llm1=llm.with_structured_output(review_sentiment)
structure_llm2=llm.with_structured_output(Diagnosis_schema)

parser=StrOutputParser()

# structure_llm1.invoke("What is the sentiment of the following review - The software too good")

# structure_llm2.invoke("What is the sentiment of the following review - The software ws too bad it is very slow in response. This is too old software wth no updates ")

class review_mail(TypedDict):
  review:str
  sentiment:Literal['positive','negative']
  diagnosis:dict
  response_mail:str

def predict_sentiment(state:review_mail):
  sentiment=structure_llm1.invoke(state['review'])
  return {'sentiment':sentiment.sentiment}

def check_sentiment(state:review_mail):

  if state['sentiment']=='positive':
    return 'positive_response'
  else:
    return 'run_diagnosis'

def positive_response(state:review_mail):
  prompt=PromptTemplate(
      template="Basis on the this review write a thank you mail and also ask for any suggestion to improve .Review:{review}",
      input_variables=['review']
  )
  chain=prompt | llm | parser
  res1=chain.invoke({'review':state['review']})
  return {'response_mail':res1}

def run_diagnosis(state:review_mail):
  diagnosis=structure_llm2.invoke(state['review'])

  return {'diagnosis':diagnosis.model_dump()}

def negative_response(state:review_mail):
    diagnosis=state['diagnosis']
    prompt=PromptTemplate(
        template="""You are a support assistant.
        The user had a {specificity} issue, sounded {tone}, and marked urgency as {urgency}.
        Write an empathetic, helpful resolution message.""",
        input_variables=['specificity','tone','urgency']
    )
    chain=prompt | llm | parser
    res1=chain.invoke({'specificity':diagnosis['specificity'],'tone':diagnosis['tone'],'urgency':diagnosis['urgency']})
    return {'response_mail':res1}

graph=StateGraph(review_mail)

## add node
graph.add_node('predict_sentiment',predict_sentiment)
graph.add_node('positive_response',positive_response)
graph.add_node('negative_response',negative_response)
graph.add_node('run_diagnosis',run_diagnosis)

## add edges
graph.add_edge(START,'predict_sentiment')
graph.add_conditional_edges('predict_sentiment',check_sentiment,{'positive_response':'positive_response','run_diagnosis':'run_diagnosis'})
graph.add_edge('positive_response',END)
graph.add_edge('run_diagnosis','negative_response')

graph.add_edge('negative_response',END)

##compile
workflow=graph.compile()


intial_state={
    'review': "I was excited about the bass and sleek design, but after 3 days, the left earbud stopped working. Customer service took 5 days to reply and offered no real solution. Save your money — these aren’t worth $50."
}

output=workflow.invoke(intial_state)

print(output['response_mail'])

print(output['diagnosis'])

print(output['sentiment'])

# new_sate={'review':"I just love using this app! It’s super fast, the UI is clean, and the login process is smooth every single time. It has really improved my productivity!"}

# res2=workflow.invoke(new_sate)

# res2['response_mail']

