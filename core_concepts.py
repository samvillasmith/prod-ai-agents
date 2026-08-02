from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model

load_dotenv()

def demo_basic_chain():
    """Demonstrates a basic chain using ChatOpenAI."""

    prompt = ChatPromptTemplate.from_template("You are a helpful assistant. Answer the following question: {question}")
    model = ChatOpenAI(model_name="gpt-4o", temperature=0)
    parser = StrOutputParser()

    # Compose with a pipe operator

    chain = prompt | model | parser

    # Run the chain with a sample question

    question = "What is the capital of France?"
    result = chain.invoke({"question": question})
    print("Result from basic chain:", result)

def demo_batch_execution():
    """Demonstrates batch processing with ChatOpenAI."""
    prompt = ChatPromptTemplate.from_template("Translate the following text to Japanese: {text}")
    model = ChatOpenAI(model_name="gpt-4o", temperature=0, max_tokens=1500, timeout=30, max_retries=3)
    parser = StrOutputParser()  

    chain = prompt | model | parser

    inputs = [{"text": "Hello, how are you?"}, {"text": "Good morning!"}, {"text": "Thank you for your help."}]
    results = chain.batch(inputs)
    
    for text, result in zip(inputs, results):
        print(f"Input: {text['text']}, Output: {result}")

def demo_streaming():
    """Demo the streaming of real-time responses from the model."""
    
    prompt = ChatPromptTemplate.from_template("You are a helpful assistant. Answer the following question: {question}")
    model = ChatOpenAI(model_name="gpt-4o", temperature=0, streaming=True)
    parser = StrOutputParser()

    chain = prompt | model | parser

    question = "Write a haiku about Zen"

    print("Streaming response:")
    for chunk in chain.stream({"question": question}):
        print(chunk, end='', flush=True)

def demo_schema_inspection():
    """Demonstration of input and output schema inspection"""
    
    prompt = ChatPromptTemplate.from_template("You are a helpful assistant. Answer the following question: {question}")
    model = ChatOpenAI(model_name="gpt-4o", temperature=0)
    parser = StrOutputParser()

    chain = prompt | model | parser

    print("Input schema:", chain.input_schema.model_json_schema())
    print("Output schema:", chain.output_schema.model_json_schema())


def exercise_first_chain():
    """Takes a product name and target audience and generate marketing tagline and returns tagline as a string
    Test with: product = "AI course" audience="developers" """

    prompt = ChatPromptTemplate.from_template(
        "You are a marketing expert. Create a catchy tagline for the product '{product}' targeting '{audience}'."
    )
    model = ChatOpenAI(model_name="gpt-4o", temperature=0)
    parser = StrOutputParser()

    chain = prompt | model | parser

    product = "AI course"
    audience = "developers"
    result = chain.invoke({"product": product, "audience": audience})
    print("Generated tagline:", result)

def new_way():
    """Universal way of invocation"""
    chat_model = init_chat_model("gpt-4o", model_provider="openai", temperature=0, max_tokens=1500)
    prompt = ChatPromptTemplate.from_template("You are a helpful assistant. Answer the following question : {question}")
    parser = StrOutputParser()

    print("Input schema:", prompt.input_schema.model_json_schema())
    print("Output schema:", parser.output_schema.model_json_schema())

    print("Invoking the chain with a sample question...")
    question = "What is the capital of Japan?"
    result = (prompt | chat_model | parser).invoke({"question": question})
    print("Result from new way:", result)



if __name__ == "__main__":
    demo_basic_chain()
    demo_batch_execution()
    demo_streaming()
    demo_schema_inspection()
    exercise_first_chain()
    new_way()