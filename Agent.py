from langgraph.graph import StateGraph, START
from typing import TypedDict, Annotated,Literal
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langgraph.types import interrupt, Command
from dotenv import load_dotenv
import requests

load_dotenv()

llm = ChatOpenAI(model="gpt-4.1-mini")
class TicketClassification(BaseModel):
    category:Literal["billing", "technical","general"]
    urgency:Literal["low","medium","high"]
    reasoning: str = Field(description="Reason for classification")

classifier_llm = llm.with_structured_output(TicketClassification)
billing_llm = ChatOpenAI(model='gpt-4o-mini')
technical_llm = ChatOpenAI(model='gpt-4o-mini')
general_llm = ChatOpenAI(model='gpt-4o-mini')

