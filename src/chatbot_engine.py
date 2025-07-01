from langchain.chat_models import ChatOpenAI
import langchain
from langchain.memory import ChatMessageHistory
from langchain.schema import HumanMessage
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo,
)
from typing import List
from langchain.tools import BaseTool
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent
from langchain.agents import AgentType

langchain.verbose = True

def create_index() -> VectorStoreIndexWrapper:
    # loader = DirectoryLoader("./src/", glob="**/*.py")
    loader = DirectoryLoader(
        "./src/", 
        glob="**/*.py", 
        loader_cls=TextLoader,  # ← ここで unstructured を使わない
        silent_errors=True       # エラーで落ちないようにする（任意）
    )
    return VectorstoreIndexCreator().from_loaders([loader])

def create_tools(index: VectorStoreIndexWrapper, llm) -> List[BaseTool]:
    vectorstore_info = VectorStoreInfo(
        vectorstore=index.vectorstore,
        name = "udemy-langchian source code",
        description = "Source code of application named udemy-langchain"
    )
    
    toolkit = VectorStoreToolkit(vectorstore_info = vectorstore_info, llm = llm)
    return toolkit.get_tools()

def chat(message: str, history: ChatMessageHistory, index: VectorStoreIndexWrapper) -> str:
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    
    tools = create_tools(index, llm)
    
    memory = ConversationBufferMemory(chat_memory = history, memory_key = "chat_history", return_messages = True)
    
    agent_chain = initialize_agent(
        tools,
        llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory
    )
    
    return agent_chain.run(message)