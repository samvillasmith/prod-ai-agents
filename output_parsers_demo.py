from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model

load_dotenv()

parser = StrOutputParser()
prompt = ChatPromptTemplate.from_template("Return a JSON object with a 'name' and 'age' field for the following description: {description}")

llm = init_chat_model(
    model="gpt-4o",
    temperature=0.7,
    max_tokens=1500,
    timeout=30,
    max_retries=3,
    model_provider="openai",
)

chain = prompt | llm | parser

result = chain.invoke({"description": "A 25-year-old developer named Chloe"})

print(result)

from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class Person(BaseModel):
    name: str = Field(..., description="The person's name")
    age: int = Field(..., description="The person's age")

parser = PydanticOutputParser(pydantic_object=Person)

prompt = ChatPromptTemplate.from_template(
    "Extract the person described below.\n{format_instructions}\nDescription: {description}"
).partial(
    format_instructions=parser.get_format_instructions()
)

chain = prompt | llm | parser
result = chain.invoke({"description": "A 19-year-old artist named Max"})
print(result)

class MovieReview(BaseModel):
    title: str = Field(..., description="The title of the movie")
    review: str = Field(..., description="A brief review of the movie")
    rating: int = Field(..., description="The rating of the movie from 1 to 10")

structured_model = llm.with_structured_output(MovieReview)

result = structured_model.invoke("Write a review for the movie 'Inception', which is a mind bending thriller with a rating of 9.")
print(result)