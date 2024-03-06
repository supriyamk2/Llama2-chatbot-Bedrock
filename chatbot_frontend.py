import streamlit as st
import chatbot_backend as demo

st.title('Hi, This is LLama 2 model')

# Load AWS credentials from Streamlit secrets
aws_access_key_id = st.secrets["supriya-pillai"]["AWS_ACCESS_KEY_ID"]
aws_secret_access_key = st.secrets["supriya-pillai"]["AWS_SECRET_ACCESS_KEY"]

print(f"AWS Access Key ID: {aws_access_key_id}")
print(f"AWS Secret Access Key: {aws_secret_access_key}")

# Initialize demo_memory with AWS credentials
if 'memory' not in st.session_state:
    st.session_state.memory = demo.demo_memory(aws_access_key_id, aws_secret_access_key)

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message['role']):
        st.markdown(message['text'])

input_text = st.chat_input('Chat with Llama 2 model here')

if input_text:
    with st.chat_message('user'):
        st.markdown(input_text)
        
    st.session_state.chat_history.append({"role": "user", "text": input_text})
    
    # Use AWS credentials from Streamlit secrets in demo_conversation
    chat_response = demo.demo_conversation(input_text=input_text, memory=st.session_state.memory,
                                           aws_access_key_id=aws_access_key_id,
                                           aws_secret_access_key=aws_secret_access_key)
    
    with st.chat_message("assistant"):
        st.markdown(chat_response)

    st.session_state.chat_history.append({"role": "assistant", "text": chat_response})
