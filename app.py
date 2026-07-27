import os
import streamlit as st
from langsmith import Client
from langchain_classic.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. Page Configuration
st.set_page_config(page_title="AI Agent", page_icon="🤖", layout="centered")
st.title("🤖 Autonomous AI Agent by Nicolas")
st.write("Powered by Google Gemini 3.5 (Free Tier). The agent uses web tools to research.")

# 2. Sidebar for API Configuration
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("Enter Google API Key", type="password")

# 3. Validation Check
if not api_key:
    st.info("Please enter your Google API Key in the sidebar to start.", icon="🔑")
    st.stop()

# Set the environment variable for Gemini to use
os.environ["GOOGLE_API_KEY"] = api_key

# 4. Initialize Agent Components
@st.cache_resource
def setup_agent():
    search_tool = DuckDuckGoSearchRun()
    tools = [search_tool]
    
    # FIX: Added max_retries=5 to automatically handle 503 high-demand errors
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash", 
        temperature=0,
        max_retries=5
    )
    
    # Fetch prompt structure
    try:
        client = Client()
        prompt = client.pull_prompt("hwchase17/openai-tools-agent")
    except Exception:
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant equipped with web search tools."),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
    
    # Construct agent executor loop
    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

try:
    agent_executor = setup_agent()
except Exception as e:
    st.error(f"Error initializing agent: {e}")
    st.stop()

# 5. User Input Interface
user_query = st.text_area(
    "What would you like the agent to do?",
    placeholder="e.g., Find out who won the latest Formula 1 world championship and multiply their total points by 2."
)

# 6. Execution and Output
if st.button("Run Agent", type="primary"):
    if not user_query.strip():
        st.warning("Please enter a prompt first.")
    else:
        with st.spinner("Agent is running. If Google's servers are busy, it will auto-retry..."):
            try:
                response = agent_executor.invoke({"input": user_query})
                st.success("Task Completed!")
                st.markdown("### Final Answer")
                st.write(response["output"])
            except Exception as e:
                st.error(f"An error occurred during execution: {e}")
