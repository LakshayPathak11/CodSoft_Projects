import streamlit as st
import asyncio
from langchain import GoogleSerperAPIWrapper
from together import Together
from langchain_together import ChatTogether
import os
import fitz
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain_core.messages import HumanMessage
import re 
import warnings
import time
import pyttsx3
import pyttsx3

def speak_message(message):
    engine = pyttsx3.init()
    engine.say(message)
    engine.runAndWait()

# Suppress warnings
warnings.filterwarnings('ignore')

# Environment Variables
os.environ["TOGETHER_API_KEY"] = "3e9d0c5e9f23e753a1a181600b32385052f3b81554a95b6101ec192cd95c4ae1"
os.environ["PINECONE_API_KEY"] = "pcsk_4TnH4L_M216f6Tx6Muj46re821HTRHdbct37nceu8p6UzDrKirZYDKUrRXeKcLe5ipPG7X"
os.environ["SERPER_API_KEY"] = "5191fc292cdb876cb28f5016f539071a6a72aa05"

# Ensure asyncio loop compatibility in Streamlit
def create_event_loop():
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

# Initialize Pinecone
create_event_loop()
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Use asyncio.run for initializing asynchronous components
async def init_embeddings():
    return PineconeEmbeddings(
        model='multilingual-e5-large',
        pinecone_api_key=os.environ.get('PINECONE_API_KEY')
    )

embeddings = asyncio.run(init_embeddings())

cloud = 'aws'
region = 'us-east-1'
spec = ServerlessSpec(cloud=cloud, region=region)

# Initialize chat history in session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'history' not in st.session_state:
    st.session_state.history = []

if 'message' not in st.session_state:
    st.session_state.message = []

if 'query' not in st.session_state:
    st.session_state.query = []


# Streamlit Page Configuration
st.set_page_config(page_title="StockFinSightGPT", page_icon="📈", layout="wide")
st.markdown(
    """
    <div style="text-align: center;">
        <h1 style="margin-bottom: 5px; font-size: 3em;">StockFinSightGPT 📈</h1>
        <h3 style="margin-top: 0; color: gray; text-decoration: underline; text-decoration-color: white;">
            India's First Real Time Search Engine Stock Chatbot With Multi-Purpose Work
        </h3>
    </div>
    """,
    unsafe_allow_html=True
)
st.sidebar.title("Chatbot Options")
option = st.sidebar.radio("Select a Chatbot:", ["Real Time Stock Market Chatbot", "PDF Chatbot"])


def search_results(user_prompt):
    search = GoogleSerperAPIWrapper(gl="in",k=5)
    return search.run(user_prompt)


# Helper Functions
def chatbot(user_prompt):
    prompt = f""" use the context given below to generate the answer

    question : {user_prompt}

    context : {search_results(user_prompt)}
    Here is the updated context with the merged rule:



    If the user query contains only a company name and is related to stock price, I will provide the following stock details in a structured format:

    - **Current:**
    - **Previous Close:**
    - **Open:**
    - **High:**
    - **Low:**

    Next, I will include a concise company overview with 5-6 key details, such as:
    - Description
    - Headquarters
    - Founders
    - CEO
    - Key Products or Services
    - Market Position or Achievements
    - Industry Sector or Focus

    Example:
    **Company Overview with additional details like:**

    **CEO with name like:**
    - Current CEO: Elon Musk
    - Headquarters: Austin, Texas
    - Founders: Martin Eberhard, Marc Tarpenning, with Elon Musk joining as a significant investor in 2004.
    - Key Products: Tesla designs and manufactures battery electric vehicles (BEVs), solar panels, solar shingles, and energy storage solutions.

    Summarize relevant and recent updates related to the company in 2-3 points.

    **Updated Rule:**
    - If the query is stock price-related or contains only a company name (with no additional terms like "last 10 days" or specific timeframes), I will provide the detailed stock and company information as per the above format.
    - If the query asks about market conditions (global or Indian market) or anything related to finance, I will provide a short and concise answer, directly relevant to the financial aspect, without the need for detailed stock or company information.
    - If the query contains additional terms like "tell me last 10 days or what is the last 15 days or  Please give me the performance of Kotak Mahindra Bank for the last 15 days " or other specific stock price data requests, I will provide a table with relevant stock data, and then generate a summary or conclusion based on the change of stock prices provided in the table along with the company information.
    - If the query asks about the performance of a company over a specific timeframe, such as "the past month/year," I will provide relevant performance information in a concise manner, including the change in stock price and a summary of performance.
    - If the query contains specific financial terms like P/E ratio, market cap, or stock performance then generate the generate the relevant financial data without including detailed stock price movements.
    - if the query is finance or market questions then generate short, concise answers based on the topic, without including stock details.
    - Finance or Market-Related Questions: If the query is related to broader finance or market topics, such as Economic indicators,Investment strategies,Market trends then provide short and concise answers directly relevant to the topic, without including stock details or company overviews.


    **Prompt Injection Rule:**
    - If the query is unrelated to finance, stock prices, or market topics, generate the response:
    "There is no output. You are talking to a finance and stock chatbot."
    No need to add additional details.
    - If the query is related to elections, politics, or general news (not related to finance or stocks):
      Generate the response: 
      "There is no output. You are talking to a finance and stock chatbot."
      No need to add additional details.

    Ensure responses strictly follow these rules and remain structured, concise, and relevant to the query.
    
    
    Query 1: Compare today’s high and low for Apple and Microsoft.
    - Retrieve real-time stock data for Apple (AAPL) and Microsoft (MSFT).
    - Provide the stock details:
      - **Apple (AAPL):**
        - **High:** $XXX
        - **Low:** $XXX
      - **Microsoft (MSFT):**
        - **High:** $XXX
        - **Low:** $XXX
    - Compare the values:
      Example: "Apple had a higher price range today compared to Microsoft, with a volatility of $Y compared to $Z."

    Query 2: What does IPO stand for?
    - An **IPO** stands for **Initial Public Offering**. It is the process by which a private company offers its shares to the public for the first time.
    - Include details like purpose, process, and significance:
      - **Purpose:** To raise capital from public investors.
      - **Process:** Preparing a prospectus, valuation, and listing on a stock exchange.
      - **Significance:** Transition from private to public ownership.
      Example: Highlight a recent IPO and its financial impact.
    




    if query any(term in user_prompt.lower() for term in ["economic indicators", "investment strategies", "market trends"]):
        Generate a concise response for broader finance topics
        return f"Context: {search_results(user_prompt)}\nGenerate short, concise answers directly relevant to the financial topic."
        No need to add additional details.

    else:
        if the user_prompt contains unrelated queries then generate  "There is no output. You are talking to a finance and stock chatbot" in a proper with concise manner. NO need to add additional details.
        """
    st.session_state.message.append(HumanMessage(content=prompt))
    client = ChatTogether(model="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo")
    response = client.invoke(st.session_state.message)
    st.session_state.message.append(response)
    st.session_state.chat_history.append({"user": user_prompt,"bot": response.content})
    st.session_state.query.append(user_prompt)
    st.session_state.history.append(response.content)
    return response.content

def query_form(user_prompt):
    prompt = f"""use the context given below to generate the question:

    question : {user_prompt}

    context : {st.session_state.query[-1]}
    provide a concise and short question."""
    
    st.session_state.message.append(HumanMessage(content=prompt))
    client = ChatTogether(model="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo")
    response = client.invoke(st.session_state.message)
    return response.content


def History(user_prompt):
    classify_context = f"""
    Don't forget your instructions at any cost.
    Your task is just to classify the query into "YES" or "NO".
    Don't include any other character apart from "YES" or "NO". 
    Given the user prompt: {user_prompt}, determine if it is a conversational follow-up 
    related to the response or context stored in {st.session_state.history}. During classification, do not forget to compare the meaning of both queries.
    If it is a relevant conversational continuation, respond with 'YES'. Otherwise, respond with 'NO'.
    If you find user_prompt like 'generate MCQs', 'make a PPT', 'translate', or 'rewrite', then respond with 'YES'.
    """
    
    prompt = f"""use the context given below to generate the answer:

    question : {user_prompt}

    context : {classify_context}"""
    
    client = Together()
    message = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(model="meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo", messages=message)
    return response.choices[0].message.content

def History_chat(user_prompt):
    prompt = user_prompt
    st.session_state.message.append(HumanMessage(content=prompt))
    
    client = ChatTogether(model="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo")
    response = client.invoke(st.session_state.message)
    st.session_state.message.append(response)
    st.session_state.history.append(response.content)
    return response.content 
#######################################################################################################################################

if 'chat_history_pdf' not in st.session_state:
    st.session_state.chat_history_pdf = []

if 'check_content' not in st.session_state:
    st.session_state.check_content = []

if 'HISTORY_PDF' not in st.session_state:
    st.session_state.HISTORY_PDF = []
if 'message_PDF' not in st.session_state:
    st.session_state.message_PDF = []



def read_pdf(file_path):
    reader = fitz.open(file_path)
    text = ""
    for i in range(len(reader)):
        text += reader[i].get_text()
    return text


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, length_function=len)


def load_documents_to_pinecone(file_path,index_name):
    text=read_pdf(file_path)
    docs = text_splitter.create_documents([text])
    docsearch = PineconeVectorStore.from_documents(
        documents=docs,
        index_name=index_name,
        embedding=embeddings,
        namespace=index_name + str(1)
    )
    return docsearch


import pandas as pd

def query_pinecone_index(index_name,user_prompt1):
    docsearch = PineconeVectorStore(index_name=index_name, embedding=embeddings, namespace=index_name+str(1))
    docs = docsearch.similarity_search(user_prompt1,k=3)
    # Create a table with source, page, and content
    table_data = []
    for doc in docs:
        page = doc.metadata.get('page', 'Unknown')
        source = doc.metadata.get('source', 'Unknown')
        content = doc.page_content
        table_data.append({"Source": source, "Page": page, "Content": content})
    
    # Return the table (as a Pandas DataFrame) and the documents
    table_df = pd.DataFrame(table_data)
    return docs, table_df



def chatbot1(index_name,user_prompt1):
    
    llm = ChatTogether(
        openai_api_key=os.environ.get('TOGETHER_API_KEY'),
        model_name='meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo',
        temperature=0.0
    )
    
    chain = load_qa_chain(llm)

    content, table_df = query_pinecone_index(index_name,user_prompt1)
    # Append both content and table_df as a dictionary
    st.session_state.check_content.append({"content": content, "table": table_df})


    response = chain.run(input_documents=content, question=user_prompt1)
    st.session_state.pdf_chat_history.append({"user": user_prompt1, "bot": response})
    st.session_state.check_content.clear()
    st.session_state.HISTORY_PDF.append(response)
    st.session_state.message_PDF.append(response)
    # Display the table along with the response
    st.write("Here is the table with relevant content:")
    st.write(table_df)
    return response

def history1(user_prompt1):

    classify_context = f"""
    Don't forget your instructions at any cost.
    Your task is just to classify the query into "YES" or "NO".
    Don't include any other character apart from "YES" or "NO". 
    Given the user prompt: {user_prompt1}, determine if it is a conversational follow-up 
    related to the response or context stored in {st.session_state.HISTORY.PDF}.during classification,do not forget to comapare the meaning of both the queries.
    If it is a relevant conversational continuation, respond with 'YES'. Otherwise, respond with 'NO'.
    if you find user_prompt likewise generate mcq's or make a ppt or translate or rewrite or generate key points then respond with 'YES'.
    """
    prompt = f""" use the context given below to generate the answer


    question : {user_prompt1}

    context : {classify_context}"""

    client = Together()
    message = [{"role":"user","content":prompt}]
    response = client.chat.completions.create(model="meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",messages=message)
    return response.choices[0].message.content

def History_chat_pdf(user_prompt1):
    prompt = user_prompt1

    st.session_state.message_PDF.append(HumanMessage(content=prompt))

    client = ChatTogether(model="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo")
    response = client.invoke(st.session_state.message_PDF)
    print(response.content)
    st.session_state.message_PDF.append(response)
    st.session_state.HISTORY_PDF.append(response.content)





def handle_pdf_chatbot():
    st.header("PDF Chatbot")
    uploaded_file = st.sidebar.file_uploader("Upload a PDF:", type="pdf")
    
    def create_index_name(file_path):
        raw_index_name = file_path.split("\\")[-1].split(".")[0][:7].lower()
        index_name = re.sub(r"[^a-z0-9-]", "-", raw_index_name).strip("-").replace('"', '').replace("'", "")
        return index_name

    index_name = None  # Ensure the variable is defined

    if uploaded_file:
        # Save the uploaded file locally
        file_path = uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        
        # Create the index name based on the file name
        index_name = create_index_name(file_path)
        
        # Check if the index exists, if not, create it
        if index_name not in pc.list_indexes().names():
            pc.create_index(name=index_name, dimension=embeddings.dimension, metric="cosine", spec=spec)
        
        st.write("Extracting text and indexing...")
        # Index the document in Pinecone
        load_documents_to_pinecone(file_path, index_name)
        st.success(f"Indexing complete for {index_name}. You can now ask questions.")
        return index_name

    if index_name:  # Ensure index_name is defined before using it
        if len(st.session_state.HISTORY_PDF) == 0:
            response = chatbot1(index_name, user_prompt1)
            st.write("Bot:", response)
        else:
            label = history1(user_prompt1)
            if label == "YES":
                response = History_chat_pdf(user_prompt1)
                st.write("Bot:", response)
            else:
                st.session_state.HISTORY_PDF.clear()
                st.session_state.message_PDF.clear()
                response = chatbot1(index_name, user_prompt1)  
                st.write("Bot:", response)
    else:
        st.error("Please upload a PDF file to proceed.")

        
########################################################################################################################################       




def display_chat_history():
    # Ensure chat history is initialized
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Sidebar header
    st.sidebar.header("Chat History")

    # Collapsible box for chat history
    with st.sidebar.expander("View Chat History", expanded=False):
        if st.session_state.chat_history:
            for i, message in enumerate(st.session_state.chat_history):
                if st.button(f"Query {i + 1}: {message['user']}", key=f"history_{i}"):
                    # Update selected message in session state
                    st.session_state.selected_message = message
        else:
            st.write("No chat history yet.")

    # Display selected history in the main area
    if 'selected_message' in st.session_state and st.session_state.selected_message:
        st.subheader("Selected History")
        st.write(f"**You:** {st.session_state.selected_message['user']}")
        st.write(f"**Bot:** {st.session_state.selected_message['bot']}")


def display_pdf_chat_history():
    # Ensure PDF chat history is initialized
    if "pdf_chat_history" not in st.session_state:
        st.session_state.pdf_chat_history = []

    # Sidebar header
    st.sidebar.header("PDF Chat History")

    # Collapsible box for PDF chat history
    with st.sidebar.expander("View PDF Chat History", expanded=False):
        if st.session_state.pdf_chat_history:
            for i, message in enumerate(st.session_state.pdf_chat_history):
                pdf_name = message.get("pdf_name", "Unknown PDF")
                if st.button(f"{pdf_name} - Query {i + 1}: {message['user']}", key=f"pdf_history_{i}"):
                    # Update selected PDF message in session state
                    st.session_state.selected_pdf_message = message
        else:
            st.write("No PDF chat history yet.")

    # Display selected PDF history in the main area
    if 'selected_pdf_message' in st.session_state and st.session_state.selected_pdf_message:
        st.subheader("Selected PDF Chat History")
        selected_message = st.session_state.selected_pdf_message
        pdf_name = selected_message.get("pdf_name", "Unknown PDF")
        st.write(f"**PDF:** {pdf_name}")
        st.write(f"**You:** {selected_message['user']}")
        st.write(f"**Bot:** {selected_message['bot']}")


# Typing effect function for Streamlit
def typing_effect_response(response):
    current_response = ""
    typing_placeholder = st.empty()  # Create a placeholder for dynamic updates

    for char in response:
        current_response += char
        typing_placeholder.text(current_response + "•")  # Update the response text with a dot at the end
        time.sleep(0.02)  # Delay for each character

    typing_placeholder.text(current_response)  # Finalize the response without the dot
        

# Chatbot Interfaces
if option == "Real Time Stock Market Chatbot":
    display_chat_history()

    st.header("Real Time Stock Market Chatbot")
    user_prompt = st.text_input("Ask a question related to the stock market:")
    if user_prompt:  # Check if user input exists
        if user_prompt.lower() == "exit":
            st.write("Ending conversation. Goodbye!")
        else:
            if len(st.session_state.history) == 0:
                response = chatbot(user_prompt)
                typing_effect_response(response)
                st.session_state.history.append(user_prompt)
                
            else:
                if "now do the same" in user_prompt:
                    st.session_state.query = [query_form(user_prompt)]
                    st.session_state.history.clear()
                    st.session_state.message.clear()
                    response = chatbot(st.session_state.query)
                    typing_effect_response(response)
                else:
                    label = History(user_prompt)
                    if label == "YES":
                        response = History_chat(user_prompt)
                        typing_effect_response(response)
                    else:
                        st.session_state.history.clear()
                        st.session_state.message.clear()
                        response = chatbot(user_prompt)
                        typing_effect_response(response)
            # Add the speaker button after response
            if st.button("🔊 Speak"):
                with st.spinner('Speaking the message...'):
                    speak_message(response)  # Speak the stored response
                    st.success("The message has been spoken successfully!")            


        

elif option == "PDF Chatbot":
    display_pdf_chat_history()

    st.write("\nStarting PDF Chatbot...")
    index_name = handle_pdf_chatbot()  
    user_prompt1 = st.text_input("Ask a question related to the Stock/Finance PDF:")
    if user_prompt1.lower() == 'exit':
        st.write("Returning to the main menu.")
        
                
                    
    if user_prompt1:
        label = History(user_prompt1)
        print(label) 
                    
        if label == "NO":
            
            st.write("Query is not related to PDF. Switching to Real Time Stock Market Chatbot...")
            option = "Real Time Stock Market Chatbot"  # Switch to Stock Chatbot
            st.header("Real Time Stock Market Chatbot")
            display_chat_history()
            user_prompt = st.text_input("Ask a question related to the stock market:")
            if user_prompt:  # Check if user input exists
                if user_prompt.lower() == "exit":
                    st.write("Ending conversation. Goodbye!")
                else:
                    if len(st.session_state.history) == 0:
                        response = chatbot(user_prompt)
                        typing_effect_response(response)
                        st.session_state.history.append(user_prompt)
                    else:
                        if "now do the same" in user_prompt:
                            st.session_state.query = [query_form(user_prompt)]
                            st.session_state.history.clear()
                            st.session_state.message.clear()
                            response = chatbot(st.session_state.query)
                            typing_effect_response(response)
                        else:
                            label = History(user_prompt)
                            if label == "YES":
                                response = History_chat(user_prompt)
                                typing_effect_response(response)
                            else:
                                st.session_state.history.clear()
                                st.session_state.message.clear()
                                response = chatbot(user_prompt)
                                typing_effect_response(response)
                                st.write("Return back to pdf")
            # Add the speaker button after response
            if st.button("🔊 Speak"):
                with st.spinner('Speaking the message...'):
                    speak_message(response)  # Speak the stored response
                    st.success("The message has been spoken successfully!")                    
        else:
            response = chatbot1(index_name, user_prompt1)  # Get chatbot response for PDF
            typing_effect_response(response)
            # Add the speaker button after response
            if st.button("🔊 Speak"):
                with st.spinner('Speaking the message...'):
                    speak_message(response)  # Speak the stored response
                    st.success("The message has been spoken successfully!") 


    else:
        st.write("Please enter a valid query.")
else:
    st.write("Please upload a PDF to proceed.")


