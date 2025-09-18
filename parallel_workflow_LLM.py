import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph,START,END
from typing import TypedDict,Literal,Annotated
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

llm=ChatGroq(
    model="llama-3.1-8b-instant"
)

class Review(TypedDict):
    product:Annotated[str,"Which product the review is about if multiple then list all seperated by comma"]
    sentiment:Literal['positive','negative']
    score:Annotated[float,"Score of the sentiment "]

structure_llm=llm.with_structured_output(Review)

prompt = """
Analyze the following product review.
Return ONLY the fields product, `sentiment` ('positive' or 'negative') 
and `score` (a float between 1 and 10).

Review: {review}
"""

template1=PromptTemplate(
    template=prompt,
    input_variables=['review']
)
review_text = """I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful"""

chain= template1 | structure_llm

output=chain.invoke({"review":review_text})
print(output)