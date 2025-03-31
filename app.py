import os
import streamlit as st
from langchain.agents import initialize_agent, AgentType
from langchain.llms import DeepSeekLLM
from langchain.tools import Tool
import yfinance as yf
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

def get_stock_price(ticker: str):
    """Fetch the latest stock price for a given ticker symbol."""
    stock = yf.Ticker(ticker)
    data = stock.history(period="1d")
    if not data.empty:
        return f"{ticker.upper()} Latest Price: ${data['Close'].iloc[-1]:.2f}"
    return "Invalid ticker or no data available."

# Initialize DeepSeek LLM using API key from .env
os.environ["DEEPSEEK_API_KEY"] = os.getenv("DEEPSEEK_API_KEY")
llm = DeepSeekLLM(model_name="deepseek-r1-distill-llama-70b")

# Define Yahoo Finance tool
yahoo_finance_tool = Tool(
    name="Stock Price Fetcher",
    func=get_stock_price,
    description="Fetches the latest stock price for a given ticker symbol."
)

# Initialize LangChain Agent
agent = initialize_agent(
    tools=[yahoo_finance_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Streamlit UI
st.title("GenAI Yahoo Finance Agent")
user_input = st.text_input("Enter stock ticker (e.g., AAPL, TSLA, GOOG):")

if st.button("Get Stock Price"):
    if user_input:
        response = agent.run(f"What is the latest stock price of {user_input}?")
        st.write(response)
    else:
        st.write("Please enter a valid stock ticker.")

# requirements.txt content
requirements = """\nos
dotenv
streamlit
langchain
yfinance
deepseek-llm
"""

with open("requirements.txt", "w") as f:
    f.write(requirements)

# Create .env file template
env_content = """\n# Add your API keys here\nDEEPSEEK_API_KEY=your-deepseek-api-key\n"""

with open(".env", "w") as f:
    f.write(env_content)