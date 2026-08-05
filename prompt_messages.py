import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model

load_dotenv()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that translates {input_language} to {output_language}."),
        ("human", "Translate the following text from {input_language} to {output_language}: {text}"),
    ]
)

messages = prompt.format_messages(
    input_language="English",
    output_language="Japanese",
    text="Hello, how are you?"
)

model = init_chat_model(
    model="gpt-4o",
    temperature=0.7,
    streaming=True,
    max_tokens=1500,
    timeout=30,
    max_retries=3,
    model_provider="openai",
)

response = model.invoke(messages)
print("Response from init_chat_model:", response.content)

### Message types

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
    ChatMessage,
    FunctionMessage,
)

# Few shot

from langchain_core.prompts import FewShotChatMessagePromptTemplate

# 1. The examples themselves. Each dict fills in one input/output pair.
examples = [
    {"input": "Hello, how are you?", "output": "こんにちは、お元気ですか？"},
    {"input": "Good morning!", "output": "おはようございます！"},
    {"input": "Thank you for your help.", "output": "ご協力ありがとうございます。"},
]

# 2. The shape of a single example: one human turn, one AI turn.
#    Use ("human", "{input}") tuples, not HumanMessage(...) — message objects
#    hold literal text and never substitute variables.
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)

# 3. Stamp that shape over every example, producing 6 messages (3 pairs).
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

# 4. Assemble the real prompt: system instructions, then the examples,
#    then the actual question. Only {text} is left to fill in.
final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that translates English to Japanese. Reply with the translation only."),
        few_shot_prompt,
        ("human", "{text}"),
    ]
)

print("******** Messages the model actually receives: ********")
for message in final_prompt.format_messages(text="Where is the train station?"):
    print(f"{message.type:>6}: {message.content}")

few_shot_response = model.invoke(final_prompt.format_messages(text="Where is the train station?"))
print("\nFew-shot response:", few_shot_response.content)
