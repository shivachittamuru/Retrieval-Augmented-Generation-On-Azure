import streamlit as st
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.azuresearch import AzureSearch

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient

service_endpoint = os.environ['AZURE_COGNITIVE_SEARCH_ENDPOINT']
key = os.environ['AZURE_COGNITIVE_SEARCH_KEY']

client = SearchIndexClient(service_endpoint, AzureKeyCredential(key))

# loading PDF, DOCX and TXT files as LangChain Documents
def load_document(file_path):
    import os
    name, extension = os.path.splitext(file_path)

    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f'Loading {file_path}')
        loader = PyPDFLoader(file_path)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f'Loading {file_path}')
        loader = Docx2txtLoader(file_path)
    elif extension == '.txt':
        from langchain.document_loaders import TextLoader
        loader = TextLoader(file_path, encoding = "utf-8", autodetect_encoding = False)
    else:
        print('Document format is not supported!')
        return None

    data = loader.load()
    return data


# splitting data in chunks
def chunk_data(data, chunk_size=256, chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks


# create embeddings using OpenAIEmbeddings() and save them in a Chroma vector store
def create_embeddings(chunks):      
    embeddings = OpenAIEmbeddings(chunk_size=1) # currently only chunk_size=1 is supported on Azure
    
    vector_store: AzureSearch = AzureSearch(
        azure_search_endpoint=os.environ['AZURE_COGNITIVE_SEARCH_ENDPOINT'],
        azure_search_key=os.environ['AZURE_COGNITIVE_SEARCH_KEY'],
        index_name="langchain-streamlit-rag-demo",
        embedding_function=embeddings.embed_query,
    )
    vector_store.add_documents(documents=chunks)
    return vector_store

def delete_index(index_name="langchain-streamlit-rag-demo"):
    client.delete_index(index_name)

def ask_and_get_answer(vector_store, question, k=3):
    from langchain.chains import RetrievalQA
    from langchain.chat_models import AzureChatOpenAI

    chat = AzureChatOpenAI(temperature=0.0,
        max_tokens=500,
        openai_api_base=os.environ['OPENAI_API_BASE'],
        openai_api_version=os.environ['OPENAI_API_VERSION'],
        deployment_name=os.environ['CHAT_MODEL_NAME'],
        openai_api_key=os.environ['OPENAI_API_KEY'],
        openai_api_type = "azure"  
    )
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})
    retrieval_chain = RetrievalQA.from_chain_type(llm=chat, chain_type="stuff", retriever=retriever)

    answer = retrieval_chain.run(question)
    return answer

def ask_with_memory(vector_store, question, chat_history=[], k=3):
    from langchain.chains import ConversationalRetrievalChain
    from langchain.chat_models import AzureChatOpenAI
    
    
    chat = AzureChatOpenAI(temperature=0.0,
        max_tokens=500,
        openai_api_base=os.environ['OPENAI_API_BASE'],
        openai_api_version=os.environ['OPENAI_API_VERSION'],
        deployment_name=os.environ['CHAT_MODEL_NAME'],
        openai_api_key=os.environ['OPENAI_API_KEY'],
        openai_api_type = "azure"  
    )
    
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=chat,
        retriever=retriever,
        chain_type="stuff",
        verbose=False
    )     
    qa_chain = ConversationalRetrievalChain.from_llm(chat, retriever)    
    result = qa_chain({"question": question, "chat_history": chat_history}) 
    chat_history.append((question, result["answer"]))
    return result['answer'], chat_history
    

# calculate embedding cost using tiktoken
def calculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    # print(f'Total Tokens: {total_tokens}')
    # print(f'Embedding Cost in USD: {total_tokens / 1000 * 0.0004:.6f}')
    return total_tokens, total_tokens / 1000 * 0.0004


# clear the chat history from streamlit session state
def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']


if __name__ == "__main__":
    import os
    
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)

    # st.image('img.png') # for logos
    st.subheader('LLM Question-Answering Application 🤖')
    with st.sidebar:
        # text_input for the OpenAI API key (alternative to python-dotenv and .env)
        # api_key = st.text_input('OpenAI API Key:', type='password')
        # if api_key:
        #     os.environ['OPENAI_API_KEY'] = api_key            
        # api_base = st.text_input('OpenAI API Base:', value='https://<aoai-service-name>.openai.azure.com')
        # if api_base:
        #     os.environ['OPENAI_API_BASE'] = api_base

        # file uploader widget
        uploaded_file = st.file_uploader('Upload a file:', type=['pdf', 'docx', 'txt'])

        # chunk size number widget
        chunk_size = st.number_input('Chunk size:', min_value=100, max_value=2048, value=512, on_change=clear_history)

        # k number input widget
        k = st.number_input('k', min_value=1, max_value=20, value=3, step=1, on_change=clear_history)               

        # add data button widget
        add_data = st.button('Add Data', on_click=clear_history)

        if uploaded_file and add_data: # if the user uploaded a file and clicked the add data button       
            # delete the index if it exists to avoid duplicate index error
            for index in client.list_index_names():
                if index=="langchain-streamlit-rag-demo":                           
                    delete_index(index)
            
            with st.spinner('Reading, chunking and embedding file ...'):                
                # # writing the file from RAM to the current directory on disk
                # bytes_data = uploaded_file.read()
                # file_name = os.path.join(f"{path}\\uploaded-files", uploaded_file.name)
                # with open(file_name, 'wb') as f:
                #     f.write(bytes_data)                    
                                                    
                path = os.getcwd()
                # data = load_document(f"{path}\\files\state_of_the_union.txt")
                data = load_document(f"{path}\\files\{uploaded_file.name}")
                chunks = chunk_data(data, chunk_size=chunk_size)
                st.write(f'Chunk size: {chunk_size}, Chunks: {len(chunks)}')

                tokens, embedding_cost = calculate_embedding_cost(chunks)
                st.write(f'Embedding cost: ${embedding_cost:.4f}')

                # creating the embeddings and returning the Azure Congitive Search vector store
                vector_store = create_embeddings(chunks)

                # saving the vector store in the streamlit session state (to be persistent between reruns)
                st.session_state.vs = vector_store
                st.success('File uploaded, chunked and embedded successfully.')

    # user's question text input widget
    q = st.text_input('Ask a question about the content of your file:')
    if q: # if the user entered a question and hit enter
        if 'vs' in st.session_state: # if there's the vector store (user uploaded, split and embedded a file)
            vector_store = st.session_state.vs
            st.write(f'k: {k}')
            
            # if there's no chat history in the session state, create it
            if 'history' not in st.session_state:
                st.session_state.history = []          
            
            answer, st.session_state.history = ask_with_memory(vector_store, q, st.session_state.history, k)

            # text area widget for the LLM answer
            st.text_area('LLM Answer: ', value=answer)

            st.divider()

            # if there's no chat history in the session state, create it
            # if 'history' not in st.session_state:
            #     st.session_state.history = ''

            # the current question and answer
            # value = f'Q: {q} \nA: {answer}'
            # st.session_state.history = f'{value} \n {"-" * 100} \n {st.session_state.history}'
            h = st.session_state.history
            st.write(f'Chat History: {h}')
            
            # text area widget for the chat history
            #st.text_area(label='Chat History', value=str(h), key='history', height=400)

# run the app: streamlit run ./rag_app.py

