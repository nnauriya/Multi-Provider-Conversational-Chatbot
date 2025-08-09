# Multi-Provider Conversational Chatbot with LangGraph

A versatile chatbot application that supports multiple AI providers and models, built with LangGraph and Streamlit. Switch seamlessly between different AI providers like Google, OpenAI, Anthropic, Groq, Cohere, and more - all in one unified interface.

## 🌟 Features

### Multi-Provider Support
- **10+ AI Providers**: Google, Groq, Cohere, Perplexity, Stability AI, OpenRouter, DeepSeek, Anthropic, Hugging Face, and ZhipuAI
- **25+ Models**: Access to various model sizes and capabilities
- **Dynamic Switching**: Change providers and models on-the-fly
- **Automatic Fallbacks**: Graceful error handling for unavailable services

### Advanced Capabilities
- **LangGraph Integration**: Robust conversation flow management
- **Message Memory**: Persistent conversation history per provider/model
- **Smart Deduplication**: Prevents duplicate consecutive messages
- **Real-time UI**: Interactive Streamlit interface
- **Image Generation**: Stability AI integration for text-to-image

## 📋 Supported Providers & Models

| Provider | Models Available | Capabilities |
|----------|------------------|--------------|
| **ZhipuAI** | glm-4.5, glm-4.5-air | Advanced Chinese/English AI |
| **Google** | gemini-2.5-flash | Fast, efficient responses |
| **Groq** | llama3-70b-8192, llama3-8b-8192 | High-speed inference |
| **Cohere** | command-r, command-light | Enterprise-grade NLP |
| **Perplexity** | sonar-pro | Web-connected AI |
| **Stability AI** | stable-diffusion-xl | Text-to-image generation |
| **OpenRouter** | 7 free models | Access to various open models |
| **DeepSeek** | deepseek-chat | Code and reasoning focused |
| **Anthropic** | claude-3-opus, claude-3-sonnet | Advanced reasoning |
| **Hugging Face** | Mistral-7B, Falcon-7B | Open source models |

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Setup
```bash
# 1. Install required packages
pip install streamlit langgraph python-dotenv

# 2. Install provider-specific packages (install as needed)
pip install langchain-google-genai      # For Google
pip install langchain-groq              # For Groq  
pip install langchain-openai            # For OpenAI/OpenRouter
pip install langchain-anthropic         # For Anthropic
pip install langchain-community         # For ZhipuAI, Perplexity, HuggingFace
pip install langchain-deepseek          # For DeepSeek
pip install cohere                      # For Cohere
pip install requests                    # For Stability AI

# 3. Create environment file
touch .env
```

### Environment Configuration
Create a `.env` file with your API keys:

```env
# Add API keys for the providers you want to use
ZHIPUAI_API_KEY=your_zhipuai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
COHERE_API_KEY=your_cohere_api_key_here
DEEPSEEK_API_KEY=your_deepseek_api_key_here
STABILITY_API_KEY=your_stability_api_key_here
PERPLEXITY_API_KEY=your_perplexity_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
```

## 🔑 Getting API Keys

### Free Options
| Provider | Free Tier | Sign Up Link |
|----------|-----------|--------------|
| **Google** | Generous free quota | [Google AI Studio](https://makersuite.google.com/app/apikey) |
| **Groq** | High free limits | [Groq Console](https://console.groq.com/) |
| **OpenRouter** | Free models available | [OpenRouter](https://openrouter.ai/) |
| **Hugging Face** | Free tier | [Hugging Face](https://huggingface.co/settings/tokens) |
| **Cohere** | Trial credits | [Cohere Dashboard](https://dashboard.cohere.com/) |

### Premium Options
| Provider | Pricing | Sign Up Link |
|----------|---------|--------------|
| **ZhipuAI** | Pay-per-use | [ZhipuAI Platform](https://open.bigmodel.cn/) |
| **Anthropic** | Usage-based | [Anthropic Console](https://console.anthropic.com/) |
| **DeepSeek** | Competitive rates | [DeepSeek Platform](https://platform.deepseek.com/) |
| **Stability AI** | Image generation credits | [Stability AI](https://platform.stability.ai/) |
| **Perplexity** | Pro subscription | [Perplexity Pro](https://www.perplexity.ai/pro) |

## 🚀 Running the Application

### Local Development
```bash
# Run the Streamlit app
streamlit run chatbot_app.py
```

The application will open in your browser at `http://localhost:8501`

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "chatbot_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t multi-provider-chatbot .
docker run -p 8501:8501 --env-file .env multi-provider-chatbot
```

## 📱 How to Use

### 1. **Select Provider & Model**
- Use the sidebar to choose your preferred AI provider
- Select from available models for that provider
- The interface shows which provider/model is active

### 2. **Start Chatting**
- Type your message in the chat input
- Press Enter or click Send
- The AI will respond using your selected model

### 3. **Switch Models**
- Change provider/model anytime in the sidebar
- Each provider/model maintains separate conversation history
- No need to restart the application

### 4. **Manage Conversations**
- **Clear Chat**: Use the sidebar button to reset current conversation
- **History**: Each provider/model combination has independent memory
- **Persistence**: Conversations persist during your browser session

## 🏗️ Architecture

### Core Components

#### State Management
```python
class State(TypedDict):
    messages: Annotated[list, add_messages]
```

#### LangGraph Workflow
```mermaid
graph LR
    A[User Input] --> B[LangGraph State]
    B --> C[Chatbot Node]
    C --> D[LLM Processing]
    D --> E[AI Response]
    E --> F[Update State]
    F --> G[Display Response]
```

#### Memory System
- **Session-based**: Conversations stored in Streamlit session state
- **Provider-specific**: Each provider/model has independent memory
- **Message validation**: Prevents duplicate consecutive messages

### Dynamic Provider Loading
The application dynamically imports and initializes LLM providers based on user selection:

```python
def load_llm(provider: str, model: str):
    # Dynamic import and initialization
    # Handles API key validation
    # Provider-specific configuration
```

## ⚙️ Configuration

### Adding New Providers
To add a new provider, update two dictionaries:

```python
# 1. Add to AVAILABLE_MODELS
AVAILABLE_MODELS["new_provider"] = ["model1", "model2"]

# 2. Add to API_KEY_MAP  
API_KEY_MAP["new_provider"] = "NEW_PROVIDER_API_KEY"

# 3. Add provider logic in load_llm function
elif provider == "new_provider":
    from langchain_new_provider import ChatNewProvider
    return ChatNewProvider(model=model, api_key=api_key)
```

### Environment Variables
Add corresponding environment variable to `.env`:
```env
NEW_PROVIDER_API_KEY=your_api_key_here
```

### Model Configuration
Customize model parameters in the provider-specific initialization:
```python
return ChatProvider(
    model=model,
    api_key=api_key,
    temperature=0.7,        # Adjust creativity
    max_tokens=1000,        # Limit response length
    # ... other parameters
)
```

## 🧪 Testing Different Providers

### Performance Comparison
Test the same query across different providers to compare:
- **Response quality**
- **Speed/latency** 
- **Creativity/style**
- **Accuracy**

### Use Case Optimization
- **Creative Writing**: Try Anthropic Claude or Google Gemini
- **Code Generation**: Test DeepSeek or Groq Llama models  
- **Fast Responses**: Use Groq for high-speed inference
- **Image Generation**: Switch to Stability AI
- **Web Search**: Use Perplexity for current information

## 📊 Provider Comparison

### Speed (Tokens/Second)
1. **Groq** - Ultra-fast inference
2. **Google Gemini** - Very fast
3. **OpenRouter Free Models** - Fast
4. **Cohere** - Moderate
5. **Anthropic** - Moderate to slow

### Capabilities
- **Best Overall**: Google Gemini, Anthropic Claude
- **Best for Code**: DeepSeek, Groq Llama
- **Best for Images**: Stability AI
- **Best Free Options**: OpenRouter, Google, Groq
- **Best Multilingual**: ZhipuAI, Google Gemini

## 🔧 Troubleshooting

### Common Issues

#### "No API key found"
- **Cause**: Missing API key in `.env` file
- **Solution**: Add the required API key for your selected provider

#### "Failed to load [provider]"
- **Cause**: Missing dependencies or invalid API key
- **Solution**: Install provider-specific packages and verify API key

#### "Import errors"
- **Cause**: Missing langchain packages
- **Solution**: Install the specific langchain package for your provider

#### Rate Limiting
- **Cause**: Exceeded API quotas
- **Solution**: Switch to a different provider or wait for quota reset

### Debug Mode
Add this to enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📈 Advanced Usage

### Conversation Management
```python
# Access conversation history
messages = st.session_state[f"{provider}_{model}_messages"]

# Clear specific provider history
del st.session_state[f"{provider}_{model}_messages"]

# Export conversation
conversation_export = [msg.content for msg in messages]
```

### Custom Message Processing
Modify the `chatbot` function to add custom logic:
```python
def chatbot(state: State):
    # Add preprocessing
    processed_messages = preprocess_messages(state["messages"])
    
    # Add custom prompting
    system_prompt = "You are a helpful assistant..."
    
    # Get response
    response = llm.invoke([system_prompt] + processed_messages)
    return {"messages": [response]}
```

### Integration Examples
```python
# Save conversations to file
def save_conversation():
    with open("conversation.json", "w") as f:
        json.dump(st.session_state[memory_key], f)

# Load external context
def add_context(context_file):
    with open(context_file, "r") as f:
        context = f.read()
    # Prepend context to messages
```

## 🚀 Deployment Options

### Streamlit Community Cloud
1. Push to GitHub repository
2. Visit [Streamlit Community Cloud](https://share.streamlit.io/)
3. Connect your repo
4. Add environment variables in app settings
5. Deploy with one click

### Railway/Render Deployment
```bash
# Railway
railway login
railway init
railway add
railway deploy

# Render
# Create render.yaml in your repo
```

### Custom Server Deployment
```bash
# Install PM2 for process management
npm install -g pm2

# Start the app
pm2 start "streamlit run chatbot_app.py --server.port 8501" --name chatbot

# Save PM2 configuration
pm2 save
pm2 startup
```

## 📁 Project Structure
```
multi-provider-chatbot/
├── chatbot_app.py          # Main application file
├── .env                    # Environment variables (create this)
├── requirements.txt        # Python dependencies  
├── README.md              # This documentation
├── Dockerfile             # Docker configuration (optional)
└── render.yaml            # Render deployment config (optional)
```

## 📦 Requirements.txt
```txt
streamlit>=1.28.0
langgraph>=0.0.30
python-dotenv>=1.0.0
langchain-core>=0.1.0
langchain-community>=0.0.20
langchain-google-genai>=1.0.0
langchain-groq>=0.1.0
langchain-openai>=0.1.0
langchain-anthropic>=0.1.0
langchain-deepseek>=0.1.0
cohere>=5.0.0
requests>=2.31.0
typing-extensions>=4.5.0
```

## 🎯 Use Cases

### Development & Testing
- **Model Comparison**: Test the same prompt across different providers
- **Performance Benchmarking**: Compare response times and quality
- **Cost Optimization**: Find the best value provider for your use case

### Production Applications
- **Chatbot Backend**: Use as a foundation for customer service bots
- **Content Generation**: Switch models based on content type
- **Research Tool**: Access multiple AI perspectives
- **Educational Platform**: Demonstrate different AI capabilities

### Specialized Tasks
- **Creative Writing**: Use creative models like Claude or Gemini
- **Code Generation**: Leverage code-focused models like DeepSeek
- **Image Creation**: Generate images with Stability AI
- **Research**: Use Perplexity for web-connected responses

## 🔒 Security Best Practices

### API Key Management
- **Never commit API keys** to version control
- **Use environment variables** for all sensitive data
- **Rotate keys regularly** for production use
- **Monitor usage** to detect unauthorized access

### Production Security
```env
# Use different keys for development and production
ENVIRONMENT=production
ZHIPUAI_API_KEY=prod_key_here
GOOGLE_API_KEY=prod_key_here
```

### Rate Limiting
- Monitor API usage across all providers
- Implement user-specific rate limiting if needed
- Set up alerts for unusual usage patterns

## 🎨 Customization

### UI Theming
```python
# Add to the top of your script
st.set_page_config(
    page_title="Custom Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.stApp > header {
    background-color: transparent;
}
</style>
""", unsafe_allow_html=True)
```

### Custom Provider Integration
```python
# Add to load_llm function
elif provider == "custom_provider":
    from custom_langchain_provider import ChatCustom
    return ChatCustom(
        model=model,
        api_key=api_key,
        custom_parameter="value"
    )
```

### Enhanced Memory
```python
# Add persistent storage
import json
import os

def save_memory():
    memory_file = f"memory_{provider}_{model}.json"
    with open(memory_file, 'w') as f:
        json.dump(st.session_state[memory_key], f)

def load_memory():
    memory_file = f"memory_{provider}_{model}.json"
    if os.path.exists(memory_file):
        with open(memory_file, 'r') as f:
            return json.load(f)
    return []
```

## 📊 Monitoring & Analytics

### Usage Tracking
```python
# Add to track usage
def log_interaction(provider, model, input_tokens, output_tokens):
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "provider": provider,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens
    }
    # Save to analytics database
```

### Performance Metrics
- **Response time per provider**
- **Token usage and costs**
- **User satisfaction ratings**
- **Error rates by provider**

## 🔧 Troubleshooting Guide

### Provider-Specific Issues

#### ZhipuAI
```bash
# Common issue: Package not found
pip install --upgrade langchain-community
```

#### Google Gemini
```bash
# Common issue: Invalid API key
# Verify key at: https://makersuite.google.com/app/apikey
```

#### Groq
```bash
# Common issue: Rate limiting
# Free tier has generous limits, check usage at console.groq.com
```

#### Stability AI
```bash
# Common issue: Image generation fails
# Verify account credits at platform.stability.ai
```

### General Troubleshooting

#### Memory Issues
```python
# Clear all provider memories
for key in list(st.session_state.keys()):
    if key.endswith("_messages"):
        del st.session_state[key]
```

#### Dependency Conflicts
```bash
# Create virtual environment
python -m venv chatbot_env
source chatbot_env/bin/activate  # Linux/Mac
# or
chatbot_env\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Advanced Features

### Conversation Export
```python
def export_conversation():
    messages = st.session_state[memory_key]
    conversation = []
    for msg in messages:
        conversation.append({
            "role": "user" if isinstance(msg, HumanMessage) else "assistant",
            "content": msg.content,
            "timestamp": datetime.now().isoformat()
        })
    return conversation
```

### Multi-Model Consensus
```python
def get_consensus_response(query, providers=["google", "groq", "cohere"]):
    responses = []
    for provider in providers:
        llm = load_llm(provider, AVAILABLE_MODELS[provider][0])
        response = llm.invoke([HumanMessage(content=query)])
        responses.append(response.content)
    return responses
```

### Conversation Analytics
```python
def analyze_conversation():
    messages = st.session_state[memory_key]
    stats = {
        "total_messages": len(messages),
        "user_messages": len([m for m in messages if isinstance(m, HumanMessage)]),
        "avg_response_length": sum(len(m.content) for m in messages) / len(messages)
    }
    return stats
```

## 📝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Test with multiple providers
5. Submit a pull request

### Adding New Providers
1. Update `AVAILABLE_MODELS` dictionary
2. Add API key mapping in `API_KEY_MAP`
3. Implement provider logic in `load_llm` function
4. Test the integration
5. Update documentation

### Code Style
- Follow PEP 8 guidelines
- Add type hints where possible
- Include docstrings for functions
- Test error handling

## 📄 License

This project is open source and available under the MIT License.

## 🆘 Support

### Getting Help
- **Issues**: Create detailed bug reports with provider/model information
- **Features**: Suggest new providers or capabilities
- **Documentation**: Request clarification on setup or usage

### Community
- Share your experience with different providers
- Contribute provider integrations
- Report performance benchmarks

---

## 🎉 Quick Start Summary

1. **Install**: `pip install streamlit langgraph python-dotenv langchain-google-genai`
2. **Configure**: Add `GOOGLE_API_KEY=your_key` to `.env` file  
3. **Run**: `streamlit run chatbot_app.py`
4. **Chat**: Select Google/gemini-2.5-flash and start chatting!

**Experience the power of multiple AI providers in one interface!** 🤖✨