"""
Sample project: Smart Q&A Bot
A production ready QA bot with structured output
"""

import os

from dataclasses import dataclass, field
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from dotenv import load_dotenv
from langsmith import traceable, Client

load_dotenv()
# langsmith configuration
if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ.setdefault("LANGSMITH_PROJECT", "Smart Q&A Bot")
    print(f"Langsmith tracing enabled: {os.getenv('LANGSMITH_TRACING')}")

Confidence = Literal["high", "medium", "low"]


class QAResponse(BaseModel):
    """Schema for the QA bot response."""

    answer: str = Field(description="The answer to the question")
    confidence: Confidence = Field(
        description="Confidence level of the answer: high, medium, or low")
    reasoning: str = Field(description="Reasoning behind the answer")
    sources_needed: Optional[bool] = Field(
        description="Indicates if sources are needed for the answer", default=False)
    follow_up_questions: List[str] = Field(
        description="List of follow-up questions for further clarification",
        default_factory=list
    )

@dataclass
class Attempt:
    """One call to one model: what came back, and what it cost."""

    model: str
    provider: str
    response: Optional[QAResponse] = None
    error: Optional[BaseException] = None
    usage: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.response is not None and self.error is None


# Implement the QA bot with structured output
class SmartQABot:
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        model_provider: str = "openai",
        temperature: Optional[float] = 0.3
    ):
        self.model_name = model_name
        self.model_provider = model_provider
        # init_chat_model keeps this class provider-agnostic, so the same bot can
        # hold an OpenAI or an Anthropic model. include_raw=True returns
        # {"raw", "parsed", "parsing_error"} instead of raising on a bad parse,
        # which is what lets the cascade below tell a failure from a weak answer.
        # temperature=None omits the parameter entirely: newer Anthropic models
        # (claude-sonnet-5 and the Opus 4.7+ line) reject sampling params with a
        # 400, so steer those with the prompt instead.
        kwargs = {} if temperature is None else {"temperature": temperature}
        self.model = init_chat_model(
            model=model_name,
            model_provider=model_provider,
            **kwargs,
        ).with_structured_output(QAResponse, include_raw=True)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a helpful assistant that answers questions with structured output.
                    your guidelines are as follows:
                    1. Provide a clear and concise answer to the user's question.
                    2. Include a confidence level (high, medium, low) based on the certainty
                    of your answer.
                    3. Provide reasoning for your answer.
                    4. Suggest follow-up questions for further clarification or exploration.
                    5. Return the response in a structured format as defined by the QAResponse schema.
                    """,
                ),
                ("human", "{question}"),
            ]
        )

        self.chain = self.prompt | self.model

    @traceable(name="attempt", run_type="chain")
    def attempt(self, question: str) -> "Attempt":
        """Ask a question and report honestly whether it worked.

        Unlike ask(), this never substitutes a fake low-confidence answer for a
        failure - the cascade needs those two cases to stay distinguishable.
        """
        try:
            result = self.chain.invoke({"question": question})
        except Exception as e:  # transport, auth, rate limit
            return Attempt(model=self.model_name, provider=self.model_provider, error=e)

        return Attempt(
            model=self.model_name,
            provider=self.model_provider,
            response=result["parsed"],
            error=result["parsing_error"],
            usage=getattr(result["raw"], "usage_metadata", None) or {},
        )

    @traceable(name="ask_question", run_type="chain")
    def ask(self, question: str) -> QAResponse:
        """Ask a question and always get a structured response back."""
        attempt = self.attempt(question)
        if attempt.ok:
            return attempt.response
        return QAResponse(
            answer="I'm sorry I couldn't process your question. Please try again.",
            confidence="low",
            reasoning=f"An error occurred while processing the question: {attempt.error}",
            follow_up_questions=[],
            sources_needed=False
        )

    @traceable(name="ask_batch", run_type="chain")
    def ask_batch(self, questions: List[str]) -> List[QAResponse]:
        """Ask a batch of questions and get structured responses.

        return_exceptions keeps one bad question from killing the whole batch.
        """
        inputs = [{"question": q} for q in questions]
        results = self.chain.batch(inputs, return_exceptions=True)

        responses: List[QAResponse] = []
        for result in results:
            if isinstance(result, Exception) or result.get("parsed") is None:
                responses.append(QAResponse(
                    answer="I'm sorry I couldn't process your question. Please try again.",
                    confidence="low",
                    reasoning=f"An error occurred while processing the question: {result}",
                ))
            else:
                responses.append(result["parsed"])
        return responses

# Example usage
def demo_qa_bot():
    """Demonstrate the Smart Q&A Bot with structured output."""
    bot = SmartQABot()

    # Single question
    question = "What is the capital of Japan?"
    response = bot.ask(question)
    print(f"Question: {question}")
    print(f"Answer: {response.answer}")
    print(f"Confidence: {response.confidence}")
    print(f"Reasoning: {response.reasoning}")
    print(f"Follow-up Questions: {response.follow_up_questions}")

    # Batch questions
    questions = [
        "What is the largest planet in our solar system?",
        "Who wrote 'Pride and Prejudice'?",
        "What is the boiling point of water?"
    ]
    batch_responses = bot.ask_batch(questions)
    for q, r in zip(questions, batch_responses):
        print(f"\nQuestion: {q}")
        print(f"Answer: {r.answer}")
        print(f"Confidence: {r.confidence}")
        print(f"Reasoning: {r.reasoning}")
        print(f"Follow-up Questions: {r.follow_up_questions}")


# ---------------------------------------------------------------------------
# Model cascading: try the cheap model first, escalate only when it falls short
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Tier:
    """A rung on the cascade. Ordered cheapest first."""

    name: str
    model: str
    provider: str
    # None means "don't send a temperature at all" - required for models that
    # reject sampling parameters.
    temperature: Optional[float] = 0.3


CHEAP = Tier("cheap", "gpt-4o-mini", "openai", temperature=0.3)
STRONG = Tier("strong", "claude-sonnet-5", "anthropic", temperature=None)


@dataclass
class CascadeResult:
    """The answer plus the audit trail of how many models it took to get it."""

    response: QAResponse
    tier: str
    attempts: List[Attempt]

    @property
    def escalated(self) -> bool:
        return len(self.attempts) > 1

    @property
    def total_tokens(self) -> int:
        return sum(a.usage.get("total_tokens", 0) for a in self.attempts)


class CascadingQABot:
    """Ask the cheap model first; escalate to the strong model only if needed.

    This is the actual cascade: the expensive model is never called unless the
    cheap one either failed or produced an answer it wasn't confident in. The
    bots are built once in __init__ rather than per question, so a long-running
    process isn't paying client construction on every call.
    """

    def __init__(
        self,
        tiers: Optional[List[Tier]] = None,
        escalate_below: Confidence = "high",
    ):
        self.tiers = tiers or [CHEAP, STRONG]
        # Anything ranked worse than escalate_below gets escalated.
        ranking = ["high", "medium", "low"]
        self.accepted = set(ranking[: ranking.index(escalate_below) + 1])
        self.bots = {
            tier.name: SmartQABot(
                model_name=tier.model,
                model_provider=tier.provider,
                temperature=tier.temperature,
            )
            for tier in self.tiers
        }

    def _good_enough(self, attempt: Attempt) -> bool:
        """Decide whether this answer ends the cascade."""
        if not attempt.ok:
            return False
        # sources_needed is the model telling us it's working from thin air,
        # which is exactly when a stronger model is worth paying for.
        if attempt.response.sources_needed:
            return False
        return attempt.response.confidence in self.accepted

    @traceable(name="cascade", run_type="chain")
    def ask(self, question: str, verbose: bool = True) -> CascadeResult:
        attempts: List[Attempt] = []

        for tier in self.tiers:
            if verbose:
                print(f"[{tier.name}] asking {tier.provider}:{tier.model} ...")

            attempt = self.bots[tier.name].attempt(question)
            attempts.append(attempt)

            if self._good_enough(attempt):
                return CascadeResult(attempt.response, tier.name, attempts)

            if verbose:
                reason = (
                    f"error: {attempt.error}" if not attempt.ok
                    else f"confidence={attempt.response.confidence}, "
                         f"sources_needed={attempt.response.sources_needed}"
                )
                print(f"[{tier.name}] not good enough ({reason})")

        # Every tier fell short. Return the best answer we actually got rather
        # than nothing; only fabricate a response if nothing succeeded at all.
        for attempt in reversed(attempts):
            if attempt.ok:
                return CascadeResult(attempt.response, "exhausted", attempts)

        errors = "; ".join(str(a.error) for a in attempts)
        return CascadeResult(
            QAResponse(
                answer="I'm sorry I couldn't process your question. Please try again.",
                confidence="low",
                reasoning=f"All tiers failed: {errors}",
            ),
            "failed",
            attempts,
        )


# ---------------------------------------------------------------------------
# Model routing: classify once, then send to exactly one model
# ---------------------------------------------------------------------------

class RouteDecision(BaseModel):
    """Schema for the routing classifier."""

    category: Literal["simple", "factual", "complex"] = Field(
        description=(
            "simple: chit-chat, formatting, or arithmetic. "
            "factual: asks for a verifiable fact that must be correct. "
            "complex: needs multi-step reasoning, analysis, or nuanced judgement."
        )
    )
    reason: str = Field(description="One sentence explaining the classification")


class QARouter:
    """Classify the question, then answer it with one model - no double spend.

    This is routing, not cascading: the classifier is a cheap dedicated call
    whose only job is to pick a lane, so the answer is generated exactly once.
    """

    def __init__(self, classifier_model: str = "gpt-4o-mini"):
        self.classifier = (
            ChatPromptTemplate.from_messages([
                ("system",
                 "Classify the user's question into exactly one category. "
                 "Do not answer it."),
                ("human", "{question}"),
            ])
            | init_chat_model(
                model=classifier_model, model_provider="openai", temperature=0
            ).with_structured_output(RouteDecision)
        )
        cheap = SmartQABot(CHEAP.model, CHEAP.provider, CHEAP.temperature)
        strong = SmartQABot(STRONG.model, STRONG.provider, STRONG.temperature)
        # factual and complex share one bot rather than building it twice.
        self.routes = {"simple": cheap, "factual": strong, "complex": strong}

    @traceable(name="route", run_type="chain")
    def ask(self, question: str, verbose: bool = True) -> QAResponse:
        decision: RouteDecision = self.classifier.invoke({"question": question})
        bot = self.routes[decision.category]

        if verbose:
            print(f"[router] {decision.category} ({decision.reason}) "
                  f"-> {bot.model_provider}:{bot.model_name}")

        return bot.ask(question)


def _print_response(response: QAResponse) -> None:
    print(f"Answer: {response.answer}")
    print(f"Confidence: {response.confidence}")
    print(f"Reasoning: {response.reasoning}")
    print(f"Follow-up Questions: {response.follow_up_questions}")


def demo_model_cascading():
    """Cheap model first, strong model only on weak or failed answers."""
    bot = CascadingQABot()
    question = input("Please enter your question: ")

    result = bot.ask(question)
    _print_response(result.response)
    print(f"\nAnswered by: {result.tier} | escalated: {result.escalated} "
          f"| models called: {len(result.attempts)} "
          f"| total tokens: {result.total_tokens}")


def demo_model_routing():
    """Classify the question once, then answer with a single model."""
    router = QARouter()
    question = input("Please enter your question: ")
    _print_response(router.ask(question))


if __name__ == "__main__":
    try:
        demo_qa_bot()
        demo_model_cascading()
    finally:
        Client().close()  # Ensure the Langsmith client is closed after execution
