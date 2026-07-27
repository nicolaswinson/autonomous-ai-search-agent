# 🤖 Autonomous Web-Searching AI Agent

An enterprise-ready AI tool runner that solves the "knowledge cutoff" limitation of standard Large Language Models. By implementing an autonomous reasoning-and-acting (ReAct) execution loop, this application empowers an LLM to dynamically browse the live internet, evaluate search fragments, and compute real-time answers.

## 🚀 Features
- **Live Web Access**: Dynamically queries the internet to answer real-time questions.
- **Robust Exception Handling**: Features automatic server retry logic (`max_retries`) to cleanly bypass temporary API network bottlenecks or service traffic spikes.
- **Interactive UI**: Built with Streamlit for a fast, responsive user interface with built-in API key masking.

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd autonomous-ai-search-agent
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   streamlit run app.py
   ```

## 🧠 Technical Architecture
This agent utilizes `langchain-classic` to orchestrate an OpenAI Tools Agent structure, passing execution logic directly to a Google Gemini model. It maintains an automated loop that detects when a user query requires outside data, formulates web queries via DuckDuckGo, and synthesizes the outputs back into a clean markdown response.
