import os
from dotenv import load_dotenv
import streamlit as st
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from typing import Annotated
from langchain_core.messages import HumanMessage, AIMessage

# Load env variables
load_dotenv()

# --- Available Providers & Models ---
AVAILABLE_MODELS = {
    **{"zhipu": ["glm-4.5", "glm-4.5-air"]},  # <-- NEW provider
    **{
        "google": ["gemini-2.5-flash"],
        "groq": ["llama3-70b-8192", "llama3-8b-8192"],
        "cohere": ["command-r", "command-light"],
        "perplexity": ["sonar-pro"],
        "stability": ["stable-diffusion-xl-1024-v1-0"],
        "openrouter": [
            "deepseek/deepseek-r1-0528:free",
            "mistralai/devstral-small-2505:free",
            "google/gemma-3n-e4b-it:free",
            "qwen/qwen3-4b:free",
            "thudm/glm-4-32b:free",
            "qwen/qwen3-32b:free",
            "moonshotai/kimi-k2:free"
        ],
        "deepseek": ["deepseek-chat"],
        "anthropic": ["claude-3-opus", "claude-3-sonnet"],
        "huggingface": [
            "mistralai/Mistral-7B-Instruct-v0.2",
            "tiiuae/falcon-7b-instruct"
        ]
    }
}

API_KEY_MAP = {
    **{"zhipu": "ZHIPUAI_API_KEY"},  # <-- NEW env
    **{
        "google": "GOOGLE_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "groq": "GROQ_API_KEY",
        "cohere": "COHERE_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "stability": "STABILITY_API_KEY",
        "perplexity": "PERPLEXITY_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "huggingface": "HUGGINGFACEHUB_API_TOKEN"
      
    }
}

# --- Sidebar UI ---
st.sidebar.title("⚙️ Settings")
provider = st.sidebar.selectbox("Select Provider", list(AVAILABLE_MODELS.keys()), index=0)
model_name = st.sidebar.selectbox("Select Model", AVAILABLE_MODELS[provider])
st.sidebar.info(f"Using {provider.upper()} | {model_name}")

# --- Dynamically Load LLM ---
def load_llm(provider: str, model: str):
    api_key = os.getenv(API_KEY_MAP.get(provider, ""))
    if not api_key:
        st.error(f"❌ No API key found for {provider.upper()} in .env")
        st.stop()

    try:
        if provider == "zhipu":
            from langchain_community.chat_models.zhipuai import ChatZhipuAI
            return ChatZhipuAI(
                model=model,
                temperature=0.7,
                api_key=api_key,
                api_base=os.getenv("ZHIPUAI_API_BASE", None)  # optional override
            )

        elif provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(model=model, google_api_key=api_key)

        elif provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=model, api_key=api_key)

        elif provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model=model, api_key=api_key)

        elif provider == "groq":
            from langchain_groq import ChatGroq
            return ChatGroq(model=model, api_key=api_key)

        elif provider == "cohere":
            import cohere
            co = cohere.Client(api_key)

            class CohereChatWrapper:
                def invoke(self, messages):
                    user_message = messages[-1].content if messages else ""
                    response = co.chat(model=model, message=user_message)
                    return AIMessage(content=response.text)
            return CohereChatWrapper()

        elif provider == "deepseek":
            from langchain_deepseek import ChatDeepSeek
            return ChatDeepSeek(model=model, api_key=api_key)

        elif provider == "stability":
            import requests
            class StabilityImageWrapper:
                def invoke(self, messages):
                    # same as your stability logic
                    prompt = messages[-1].content if messages else "A futuristic AI robot"
                    headers = {
                        "Accept": "application/json",
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key}"
                    }
                    payload = {
                        "text_prompts": [{"text": prompt}],
                        "cfg_scale": 7, "samples": 1, "steps": 30
                    }
                    response = requests.post(
                        "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image",
                        headers=headers, json=payload
                    )
                    if response.status_code == 200:
                        img_base64 = response.json()["artifacts"][0]["base64"]
                        img_data = f"data:image/png;base64,{img_base64}"
                        return AIMessage(content=f"✅ Image generated:\n![Image]({img_data})")
                    return AIMessage(content=f"❌ Image generation failed: {response.text}")
            return StabilityImageWrapper()

        elif provider == "perplexity":
            from langchain_community.chat_models import ChatPerplexity
            return ChatPerplexity(model=model, api_key=api_key)

        elif provider == "openrouter":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=model, api_key=api_key, base_url="https://openrouter.ai/api/v1")

        elif provider == "huggingface":
            from langchain_community.llms import HuggingFaceEndpoint
            return HuggingFaceEndpoint(repo_id=model, huggingfacehub_api_token=api_key, task="text-generation")

        else:
            raise ValueError(f"Provider '{provider}' not supported!")

    except Exception as e:
        st.error(f"❌ Failed to load {provider}: {str(e)}")
        st.stop()

llm = load_llm(provider, model_name)

# --- LangGraph Setup ---
class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

def chatbot(state: State):
    valid_messages = []
    last_role = None
    for msg in state["messages"]:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        if role == last_role:
            continue
        valid_messages.append(msg)
        last_role = role
    return {"messages": [llm.invoke(valid_messages)]}

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

# --- Memory Key ---
memory_key = f"{provider}_{model_name}_messages"
if memory_key not in st.session_state:
    st.session_state[memory_key] = []

# --- Streamlit UI ---
st.set_page_config(page_title="Multi-Provider Conversational Chatbot", page_icon="🤖")
st.title(f"🤖 {provider.upper()} | {model_name} Chatbot")

for msg in st.session_state[memory_key]:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state[memory_key].append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    state = {"messages": st.session_state[memory_key]}
    result = graph.invoke(state)
    ai_msg = result["messages"][-1]
    st.session_state[memory_key].append(ai_msg)

    with st.chat_message("assistant"):
        st.markdown(ai_msg.content)

if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state[memory_key] = []
    st.experimental_rerun()
