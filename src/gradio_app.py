from dotenv import load_dotenv
from chatbot_engine import chat, create_index
from langchain.memory import ChatMessageHistory
import gradio as gr

import os

def respond(message, chat_history):
        history = ChatMessageHistory()
        
        for [user_message, ai_message] in chat_history:
            history.add_user_message(user_message)
            history.add_ai_message(ai_message)
        
        bot_message = chat(message, history, index)
        chat_history.append((message, bot_message))
        return "", chat_history
    
with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")
    
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, chatbot, queue=False)
    
if __name__ == "__main__":
    load_dotenv()
    
    app_env = os.environ.get("APP_ENV", "production")
    
    if app_env == "production":
        user_name = os.environ["GRADIO_USERNAME"]
        password = os.environ["GRADIO_PASSWORD"]
        
        auth = (user_name, password)
    else:
        auth = None
        
    index = create_index()
    demo.launch(auth = auth, server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 8080)))