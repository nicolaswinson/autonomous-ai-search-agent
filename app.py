import os
import io
import json
import base64

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend, required inside Streamlit
import matplotlib.pyplot as plt

import streamlit as st
from langsmith import Client
from langchain_classic.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

# ----------------------------------------------------------------------------
# 1. Page configuration
# ----------------------------------------------------------------------------
st.set_page_config(page_title="AI Agent", page_icon="🤖", layout="centered")
st.title("🤖 Autonomous AI Agent by Nicolas")
st.write(
    "Powered by Google Gemini. The agent can search the web, remember the "
    "conversation, draw charts from data, and generate images."
)

# ----------------------------------------------------------------------------
# 2. Sidebar — configuration
# ----------------------------------------------------------------------------
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("Enter Google API Key", type="password")

text_model = st.sidebar.selectbox(
    "Reasoning model",
    ["gemini-3.5-flash", "gemini-3.1-pro-preview"],
    index=0,
    help="Flash is faster/cheaper, Pro is stronger at multi-step reasoning.",
)

if st.sidebar.button("🗑️ Clear conversation"):
    st.session_state.messages = []
    st.session_state.lc_history = []
    st.rerun()

# ----------------------------------------------------------------------------
# 3. Validation check
# ----------------------------------------------------------------------------
if not api_key:
    st.info("Please enter your Google API Key in the sidebar to start.", icon="🔑")
    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key

# ----------------------------------------------------------------------------
# 4. Session state
# ----------------------------------------------------------------------------
# `messages` drives what's rendered on screen (text + any images/charts).
# `lc_history` is the trimmed LangChain message list fed back into the agent
# as memory on every turn.
if "messages" not in st.session_state:
    st.session_state.messages = []
if "lc_history" not in st.session_state:
    st.session_state.lc_history = []
# Scratch space that tool calls drop artifacts into during a single run.
if "_pending_images" not in st.session_state:
    st.session_state._pending_images = []
if "_pending_charts" not in st.session_state:
    st.session_state._pending_charts = []

MAX_HISTORY_MESSAGES = 12  # keep the last N turns of memory (context control)


# ----------------------------------------------------------------------------
# 5. Tools
# ----------------------------------------------------------------------------
search_tool = DuckDuckGoSearchRun()


@tool
def generate_image(prompt: str) -> str:
    """Generate an image from a text description and show it to the user.
    Use this whenever the user asks to draw, create, generate, or illustrate
    a picture/image/photo/logo/icon of something. `prompt` should be a clear,
    descriptive image prompt in English."""
    try:
        from google import genai as google_genai

        client = google_genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=[prompt],
        )

        found = False
        for part in response.candidates[0].content.parts:
            inline_data = getattr(part, "inline_data", None)
            if inline_data is not None and inline_data.data:
                st.session_state._pending_images.append(inline_data.data)
                found = True

        if not found:
            return "The image model did not return any image data for that prompt."
        return f"Image generated successfully for prompt: '{prompt}'."
    except Exception as e:
        return f"Image generation failed: {e}"


@tool
def create_chart(spec: str) -> str:
    """Create a data visualization (bar, line, scatter, or pie chart) and show
    it to the user. `spec` must be a JSON string with the fields:
    {
      "chart_type": "bar" | "line" | "scatter" | "pie",
      "title": "chart title",
      "x_label": "x axis label" (optional),
      "y_label": "y axis label" (optional),
      "data": [{"label": "A", "value": 10}, {"label": "B", "value": 20}, ...]
    }
    Use this whenever the user asks to plot, chart, graph, or visualize data."""
    try:
        payload = json.loads(spec)
        chart_type = payload.get("chart_type", "bar").lower()
        title = payload.get("title", "")
        x_label = payload.get("x_label", "")
        y_label = payload.get("y_label", "")
        df = pd.DataFrame(payload["data"])

        fig, ax = plt.subplots(figsize=(6, 4))
        if chart_type == "bar":
            ax.bar(df["label"], df["value"])
        elif chart_type == "line":
            ax.plot(df["label"], df["value"], marker="o")
        elif chart_type == "scatter":
            ax.scatter(df["label"], df["value"])
        elif chart_type == "pie":
            ax.pie(df["value"], labels=df["label"], autopct="%1.1f%%")
        else:
            return f"Unsupported chart_type '{chart_type}'. Use bar, line, scatter, or pie."

        ax.set_title(title)
        if chart_type != "pie":
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            plt.xticks(rotation=30, ha="right")
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        buf.seek(0)
        st.session_state._pending_charts.append(buf.getvalue())

        return f"Chart '{title}' ({chart_type}) created successfully."
    except Exception as e:
        return f"Chart creation failed: {e}. Make sure `spec` is valid JSON matching the schema."


tools = [search_tool, generate_image, create_chart]


# ----------------------------------------------------------------------------
# 6. Agent setup
# ----------------------------------------------------------------------------
@st.cache_resource
def setup_agent(model_name: str):
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0, max_retries=5)

    try:
        client = Client()
        prompt = client.pull_prompt("hwchase17/openai-tools-agent")
    except Exception:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant equipped with web search, "
                    "image generation, and data visualization tools. Use "
                    "chat_history to remember what the user already told you.",
                ),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


try:
    agent_executor = setup_agent(text_model)
except Exception as e:
    st.error(f"Error initializing agent: {e}")
    st.stop()


def extract_text(raw_output) -> str:
    """Normalize Gemini's occasionally block-shaped output into plain text."""
    if isinstance(raw_output, list) and raw_output:
        first = raw_output[0]
        clean = first.get("text", str(first)) if isinstance(first, dict) else str(first)
    elif isinstance(raw_output, dict):
        clean = raw_output.get("text", str(raw_output))
    else:
        clean = str(raw_output)
    return clean.replace("\\n", "\n")


# ----------------------------------------------------------------------------
# 7. Render chat history
# ----------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        for img_bytes in msg.get("images", []):
            st.image(img_bytes)
        for chart_bytes in msg.get("charts", []):
            st.image(chart_bytes)

# ----------------------------------------------------------------------------
# 8. Chat input and execution
# ----------------------------------------------------------------------------
user_query = st.chat_input(
    "Ask the agent anything — e.g. 'chart Q1-Q4 sales of 10, 15, 9, 20' "
    "or 'generate an image of a robot reading a book'"
)

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    st.session_state._pending_images = []
    st.session_state._pending_charts = []

    with st.chat_message("assistant"):
        with st.spinner("Agent is thinking and executing tools..."):
            try:
                response = agent_executor.invoke(
                    {
                        "input": user_query,
                        "chat_history": st.session_state.lc_history[-MAX_HISTORY_MESSAGES:],
                    }
                )
                clean_output = extract_text(response.get("output", ""))

                st.markdown(clean_output)
                new_images = list(st.session_state._pending_images)
                new_charts = list(st.session_state._pending_charts)
                for img in new_images:
                    st.image(img)
                for chart in new_charts:
                    st.image(chart)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": clean_output,
                        "images": new_images,
                        "charts": new_charts,
                    }
                )
                st.session_state.lc_history.append(HumanMessage(content=user_query))
                st.session_state.lc_history.append(AIMessage(content=clean_output))

            except Exception as e:
                error_msg = f"An error occurred during execution: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
