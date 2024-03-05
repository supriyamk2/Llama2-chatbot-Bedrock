import os
from langchain_community.llms import Bedrock
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

def demo_chatbot():
    aws_access_key_id = st.secrets["AWS_ACCESS_KEY_ID"]
    aws_secret_access_key = st.secrets["AWS_SECRET_ACCESS_KEY"]

    demo_llm = Bedrock(
        credentials_profile_name='default',  # Adjust profile name if needed
        model_id='meta.llama2-70b-chat-v1',
        model_kwargs={
            "temperature": 0.9,
            "top_p": 0.5,
            "max_gen_len": 512
        }, 
        client_kwargs={  
           "aws_access_key_id": aws_access_key_id,
           "aws_secret_access_key": aws_secret_access_key,
       }
    )
    return demo_llm

    #return demo_llm.invoke(input_text)


#response = demo_chatbot("hi, what is your name?")
#print(response)

def demo_memory():
    llm_data = demo_chatbot()
    memory = ConversationBufferMemory(llm = llm_data, max_token_limit = 512)
    return memory

def demo_conversation(input_text, memory):
    llm_chain_data = demo_chatbot()
    llm_conversation = ConversationChain(llm=llm_chain_data, memory=memory, verbose=True)
    
    chat_reply = llm_conversation.predict(input = input_text)
    return chat_reply
    


    






