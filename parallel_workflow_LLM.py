import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph,START,END
from typing import TypedDict,Literal,Annotated
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm=ChatGroq(
    model="llama-3.1-8b-instant"
)
parser=StrOutputParser()

class Review(TypedDict):
    review_detail:Annotated[str,"Reviews of the product"]
    product:Annotated[str,"Which product the review is about if multiple then list all seperated by comma"]
    sentiment:Literal['positive','negative']
    # score:Annotated[float,"Score of the sentiment between 1 and 10"]
    summary:Annotated[str,'name and Sentimen of the product']

structure_llm=llm.with_structured_output(Review)

prompt = """
Analyze the following product review.
Return ONLY the fields product, `sentiment` ('positive' or 'negative') 
and `score` (a float between 1 and 10).

Review: {review}
"""

# template1=PromptTemplate(
#     template=prompt,
#     input_variables=['review']
# )
review_text = """I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful"""

## product_extract
def product_extract(state:Review):
    review=state['review_detail']
    prompt1 = """
    Analyze the following product review.
    Return ONLY the field product name: 

    Review: {review}
    """
    template1=PromptTemplate(template=prompt1,input_variables=['review'])
    chain1=template1 | llm | parser
    output=chain1.invoke({'review':review_text})
    return {'product':output}

def sentiment_review(state:Review):
    review=state['review_detail']
    prompt1 = """
    Analyze the following product review.
    Return ONLY the field sentiment of the product

    Review: {review}
    """
    template2=PromptTemplate(template=prompt1,input_variables=['review'])
    chain2=template2 | llm | parser
    output2=chain2.invoke({'review':review_text})
    return {'sentiment':output2}

def summary(state:Review):
    summary_report=f"The name of product is {state['product']}, and sentiment regarding product is {state['sentiment']}"
    return {'summary':summary_report}

## define graph
graph=StateGraph(Review)

## add node
graph.add_node('product_extract',product_extract)
graph.add_node('sentiment_review',sentiment_review)
graph.add_node('summary',summary)

## add edges
graph.add_edge(START,'product_extract')
graph.add_edge(START,'sentiment_review')
graph.add_edge('product_extract','summary')
graph.add_edge('sentiment_review','summary')
graph.add_edge('summary',END)

## compile graph
workflow=graph.compile()

## exectue graph
initital_state={'review_detail':review_text}

result=workflow.invoke(initital_state)
print(result['summary'])


# chain= template1 | structure_llm

# output=chain.invoke({"review":review_text})
# print(output)