from dotenv import load_dotenv
load_dotenv()

from importlib.metadata import version

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

print("langchain-core version:", version("langchain-core"))
print("langgraph version:", version("langgraph"))
print("langchain-openai:", ChatOpenAI)
print("langchain-anthropic:", ChatAnthropic)

def main():
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    response = llm.invoke("Hello, how are you?")
    print("Response from ChatOpenAI:", response)

    llm_anthropic = ChatAnthropic(model_name="claude-haiku-4-5", temperature=0)
    response_anthropic = llm_anthropic.invoke("Hello, how are you?")
    print("Response from ChatAnthropic:", response_anthropic)

    print("Setup complete. You can now use the langchain-course package.")


if __name__ == "__main__":
    main()