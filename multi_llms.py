"""Working with multiple LLMs in V1. 
Multiple providers, configuration, streaming, and cost optimization.
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

def demo_init_chat_model():
    chat_model = init_chat_model(
        model="gpt-4o",
        temperature=0.7,
        streaming=True,
        max_tokens=1500,
        timeout=30,
        max_retries=3,
        model_provider="openai",
    )

    response = chat_model.invoke([SystemMessage(content="You are a helpful assistant."), HumanMessage(content="What is the capital of Japan?")])

    print("Response from init_chat_model:", response.content)

    if os.getenv("ANTHROPIC_API_KEY"):
        anthropic_model = init_chat_model(
            model="claude-haiku-4-5",
            temperature=0.7,
            streaming=True,
            max_tokens=1500,
            timeout=30,
            max_retries=3,
            model_provider="anthropic",
        )

        response_anthropic = anthropic_model.invoke([SystemMessage(content="You are a helpful assistant."), HumanMessage(content="What is the capital of Japan?")])

        print("Response from Anthropic model:", response_anthropic.content)

def demo_model_comparison():
    prompt = "Explain the theory of relativity in simple terms in one sentence."

    specs = [("openai", "gpt-4o")]
    if os.getenv("ANTHROPIC_API_KEY"):
        specs.append(("anthropic", "claude-haiku-4-5"))

    for provider, name in specs:
        model = init_chat_model(model=name, temperature=0.7, model_provider=provider)
        response = model.invoke([SystemMessage(content="You are a helpful assistant."), HumanMessage(content=prompt)])
        print(f"Response from {provider} ({name}): {response.content}")

def demo_message():
    model = ChatOpenAI(model_name="gpt-4o", temperature=0.7)
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="What is the most prevalent religion in Japan?"),
    ]
    response = model.invoke(messages)
    messages.append(response)
    print("----------------------------------------------------------------")
    print("Full messages list contents:", messages)

def multi_model():
    """Example:
    1. Take a question and a list of model names
    2. Get a response from each model
    3. Return the responses in a dictionary
    """
    models = ["gpt-4o", "claude-haiku-4-5"]
    question = "What is the most underrated place to visit in Japan?"
    responses = {}  

    print("****** Responses from multiple models ******")

    for model_name in models:
        if model_name.startswith("gpt"):
            model = ChatOpenAI(model_name=model_name, temperature=0.7)
        elif model_name.startswith("claude"):
            model = init_chat_model(model=model_name, temperature=0.7, model_provider="anthropic")
        else:
            continue

        response = model.invoke([SystemMessage(content="You are a helpful assistant."), HumanMessage(content=question)])
        responses[model_name] = response.content

    print(responses)

    

if __name__ == "__main__":
    demo_init_chat_model()
    demo_model_comparison()
    demo_message()
    multi_model()