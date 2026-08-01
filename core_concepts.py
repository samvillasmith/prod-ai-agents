from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

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
    model = ChatOpenAI(model_name="gpt-4o", temperature=0)
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

if __name__ == "__main__":
    demo_basic_chain()
    demo_batch_execution()
    demo_streaming()