
from langgraph.graph import StateGraph, START , END
from typing import TypedDict, Annotated,Literal
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel, Field
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langgraph.types import interrupt, Command
from dotenv import load_dotenv
from langchain_core.tools import tool
from langgraph.types import interrupt
from langgraph.checkpoint.memory import InMemorySaver
import requests
import re


load_dotenv()

llm = ChatOpenAI(model="gpt-4.1-mini" , temperature=0 , timeout= 60 , max_retries=1)

billing_llm = ChatOpenAI(model='gpt-4.1-mini')
technical_llm = ChatOpenAI(model='gpt-4.1-mini')
general_llm = ChatOpenAI(model='gpt-4.1-mini')
reviewer_llm = ChatOpenAI(model='gpt-4.1-mini')

@tool
def extract_order_id(message: str) -> str:
    """
    Extracts an order ID from a customer's support message
    using regular expressions.
    """

    pattern = r"(?:order\s*#?\s*|#)([A-Za-z0-9-]+)"

    match = re.search(pattern, message, re.IGNORECASE)

    if match:
        return match.group(1)

    return "No order ID found"


class ReviewResult(BaseModel):
    score: int = Field(
        description="Quality score between 0 and 10",
        ge=0,
        le=10
    )

    feedback: str = Field(
        description="Specific improvements needed in the response"
    )


# Use your existing LLM with structured output
reviewer_llm = billing_llm.with_structured_output(ReviewResult)

class TicketClassification(BaseModel):
    category:Literal["billing", "technical","general"]
    urgency:Literal["low","medium","high"]
    reasoning: str = Field(description="Reason for classification")

classifier_llm = llm.with_structured_output(TicketClassification)

class TicketState(TypedDict):
    ticket:str
    category:str
    urgency:str
    reasoning:str
    response:str

    review_score:str
    review_feedback:str
    attempts:int


#classifying the incoming state
def classify_ticket(state:TicketState):
    result = classifier_llm.invoke(f"""
        Classify this customer support ticket.

        Categories:
        - billing: payments, refunds, charges, orders
        - technical: bugs, crashes, login or app issues
        - general: other questions

        Assign urgency as low, medium, or high.

        Ticket: {state['ticket']}
        """)
    return{
        "category":result.category,
        "urgency":result.urgency,
        "reasoning" : result.reasoning
    }

def route_ticket(state:TicketState) ->str:
    category = state['category']

    if category == "billing":
        return "billing_handler"
    elif category == "technical":
        return "technical_handler"
    else:
        return "general_handler"

def reviewer(state: TicketState):

    review = reviewer_llm.invoke(f"""
        You are an expert customer support response reviewer.

        Evaluate the following response.

        Customer ticket: {state['ticket']}

        Generated response: {state['response']}

        Urgency: {state['urgency']}

        Evaluate:
        1. Relevance to the customer query.
        2. Accuracy and factual consistency.
        3. Clarity and professionalism.
        4. Whether it makes unsupported claims.
        Give a score from 0 to 10.
        If the response is poor, explain exactly
        what should be improved.
    """)

    attempts = min(state.get("attempts", 0) + 1, 3)

    print(f"Review attempt: {attempts}")
    print(f"Score: {review.score}/10")
    print(f"Feedback: {review.feedback}")

    return {
        "review_score": review.score,
        "review_feedback": review.feedback,
        "attempts": attempts
    }


def billing_handler(state:TicketState):
    order_id = extract_order_id.invoke({
        "message": state["ticket"]
    })
    feedback = state.get("review_feedback", "")
    result = billing_llm.invoke(f"""
        You are a customer billing support agent.
        Help with charges, refunds, and payments.
        Do not claim a refund or payment action was completed
        unless it has actually been verified.
        Return one customer-facing reply only. Do not repeat it,
        add a sign-off, placeholder name, or department signature.
        Customer ticket: {state['ticket']}
        Urgency: {state['urgency']}
        Extracted Order ID: {order_id}
        Previous reviewer feedback: {feedback}

        Improve your response using the reviewer's
        feedback. If there is no previous feedback,
        generate the best possible response.
        """)
    return {"response": result.content}


def human_approval(state: TicketState):

    decision = interrupt({
        "message": "Billing response requires human approval",
        "options": ["Yes", "No"],
        "customer_ticket": state["ticket"],
        "draft_response": state["response"],
        "instruction": (
            "Choose Yes to approve the draft. Choose No to reject it "
            "and provide a corrected final reply."
        )
    })

    if decision.get("approved") is True:
        final_response = state["response"]
    elif decision.get("approved") is False and decision.get("edited_response", "").strip():
        final_response = decision["edited_response"].strip()
    else:
        raise ValueError("A rejected billing draft requires a corrected reply.")

    return {
        "response": final_response
    }


def final_reply(state: TicketState):
    response = state["response"].strip()
    response = re.sub(
        r"(?is)\n*(?:best regards|kind regards|regards|sincerely)\b.*$",
        "",
        response,
    ).strip()
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", response) if part.strip()]
    response = "\n\n".join(dict.fromkeys(paragraphs))
    print(response)
    return {"response": response}




def review_router(state: TicketState):

    # Review and attempts:
    if state["review_score"] < 8 and state["attempts"] < 3:
        return "classify_ticket"

    # Billing tickets require human approval
    if state["category"] == "billing":
        return "human_approval"

    # Other tickets skip human approval
    return "final_reply"



def technical_handler(state:TicketState):
    result = technical_llm.invoke(f"""
        You are a technical support agent.
        Help customers troubleshoot app errors and technical issues.
        Give clear, practical steps.
        Return one customer-facing reply only. Do not repeat it,
        add a sign-off, placeholder name, or department signature.

        Customer ticket: {state['ticket']}
        Urgency: {state['urgency']}
        """)
    return {"response": result.content}


def general_handler(state: TicketState):
    result = general_llm.invoke(
        f"""
        You are a general customer support agent.
        Answer politely, clearly, and accurately.
        Return one customer-facing reply only. Do not repeat it,
        add a sign-off, placeholder name, or department signature.

        Customer ticket: {state['ticket']}
        Urgency: {state['urgency']}
        """
    )
    return {"response": result.content}


graph = StateGraph(TicketState)

# Register nodes
graph.add_node("classify_ticket", classify_ticket)
graph.add_node("billing_handler", billing_handler)
graph.add_node("technical_handler", technical_handler)
graph.add_node("general_handler", general_handler)
graph.add_node("reviewer", reviewer)
graph.add_node("human_approval", human_approval)
graph.add_node("final_reply", final_reply)

# Reset review state for each new run, including reused thread IDs
graph.add_node(
    "initialize_review",
    lambda state: {"attempts": 0, "review_score": 0, "review_feedback": ""}
)
graph.add_edge(START, "initialize_review")
graph.add_edge("initialize_review", "classify_ticket")

# Classification routing
graph.add_conditional_edges(
    "classify_ticket",
    route_ticket,
    {
        "billing_handler": "billing_handler",
        "technical_handler": "technical_handler",
        "general_handler": "general_handler"
    }
)

# All handlers go to reviewer
graph.add_edge("billing_handler", "reviewer")
graph.add_edge("technical_handler", "reviewer")
graph.add_edge("general_handler", "reviewer")

# Reviewer routing
graph.add_conditional_edges(
    "reviewer",
    review_router,
    {
        "classify_ticket": "classify_ticket",
        "human_approval": "human_approval",
        "final_reply": "final_reply"
    }
)

# Human approval to final reply
graph.add_edge("human_approval", "final_reply")


# End
graph.add_edge("final_reply", END)

# Checkpointer is required for interrupt/resume
memory = InMemorySaver()

workflow = graph.compile(checkpointer=memory)


workflow

config = {"configurable": {"thread_id": "test_1"}}

# Run until it pauses for human approval
workflow.invoke({"ticket": "My money has been deducted from my account but the ticket is not confirmed , #1087."}, config=config)

# Auto-approve and finish
result = workflow.invoke(Command(resume={"approved": True}),config=config)

print("Category:",result["category"])
print("Review_Score:",result["review_score"])
print("Response:",result["response"])

config = {"configurable": {"thread_id": "test_tech_1"}}

result = workflow.invoke(
    {"ticket": "The app keeps crashing every time I try to log in."},config=config)

print("Category:", result["category"])
print("Review_Score:", result["review_score"])
print("Response:", result["response"])

config = {"configurable": {"thread_id": "test_general_1"}}

result = workflow.invoke(
    {"ticket": "What are your customer support working hours?"},
    config=config
)

print("Category:", result["category"])
print("Review_Score:", result["review_score"])
print("Response:", result["response"])
