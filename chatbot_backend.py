import os
from langchain_community.llms import Bedrock
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

def demo_chatbot(aws_access_key_id, aws_secret_access_key):
    client_kwargs = {
        "aws_access_key_id": aws_access_key_id,
        "aws_secret_access_key": aws_secret_access_key,
    }

    demo_llm = Bedrock(
        credentials_profile_name='default',
        model_id='meta.llama2-70b-chat-v1',
        model_kwargs={
            "temperature": 0.9,
            "top_p": 0.5,
            "max_gen_len": 512
        },
        client_kwargs=client_kwargs  # Use the modified client_kwargs
    )

    return demo_llm


# Pass AWS credentials to demo_memory
def demo_memory(aws_access_key_id, aws_secret_access_key):
    llm_data = demo_chatbot(aws_access_key_id, aws_secret_access_key)
    memory = ConversationBufferMemory(llm=llm_data, max_token_limit=512)
    return memory

# Pass AWS credentials and memory to demo_conversation
def demo_conversation(input_text, aws_access_key_id, aws_secret_access_key, memory):
    llm_chain_data = demo_chatbot(aws_access_key_id, aws_secret_access_key)
    llm_conversation = ConversationChain(llm=llm_chain_data, memory=memory, verbose=True)
    
    chat_reply = llm_conversation.predict(input=input_text)
    return chat_reply
