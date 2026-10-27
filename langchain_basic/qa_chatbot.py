import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

## Page Config
st.set_page_config(
    page_title='Simple Langchain Chatbot with Groq',
    page_icon=''
)

st.title('Simple Langchain Chatbot with Groq')

with st.sidebar:
    st.header('Settings')
    # API key
    api_key = st.text_input(
        'Please enter Groq API key', 
        type='password', 
        help='Get free API key at console.groq.com'
    )

    # Model Selection
    model_list = [
        'openai/gpt-oss-120b',
        'meta-llama/llama-prompt-guard-2-22m',
        'openai/gpt-oss-20b',
        'qwen/qwen3.8-27b',
    ]

    model = st.selectbox('Please select a model', model_list, index=0)

    # clear button
    if st.button('Clear Chat'):
        st.session_state.messages = []
        st.rerun()

# Initialize Chat History
if 'messages' not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def get_chain(api_key, model_name):
    if not api_key:
        return None

    # Initialize Groq Model
    llm = ChatGroq(model=model_name, api_key=api_key, temperature=0.7, streaming=True)

    # Create Prompt Template
    prompt = ChatPromptTemplate.from_messages([
        ('system', '''
        You are an helpful AI Assistant Powered by Groq.
        Answer question clearly and concisely.
        '''),
        ('user', '{question}')
    ])

    # Create Chain and Return 
    chain = prompt | llm | StrOutputParser()
    return chain

chain = get_chain(api_key=api_key, model_name=model)

if not chain:
    st.warning('Please enter your Groq API key in sidebar')
    st.markdown('[Get your free API key here](https://console.groq.com)')
else:
    # Display the chat message
    for message in st.session_state.messages:
        role = 'user' if isinstance(message, HumanMessage) else 'assistant'
        with st.chat_message(role):
            st.markdown(message.content)

question = st.chat_input('Ask me anything...') if chain else None

if question:
    # Add user message to session state
    st.session_state.messages.append(HumanMessage(content=question))

    # Display user message
    with st.chat_message('user'):
        st.markdown(question)

    # Generate response 
    with st.chat_message('assistant'):
        message_placeholder = st.empty()
        full_response = ''
        try:
            # Stream response from Groq
            for chunk in chain.stream({'question': question}):
                full_response += chunk
                message_placeholder.markdown(full_response + '|')

            # message_placeholder.markdown(full_response)
            
            # Add to session history
            st.session_state.messages.append(AIMessage(content=full_response))
        except Exception as e:
            st.error(f'Error: {str(e)}')

## Footer
st.markdown('---')
st.markdown('Built with Langchain & Groq | Experince the speed!')
