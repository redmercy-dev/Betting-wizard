import streamlit as st
import openai
import os
from llama_index.agent import OpenAIAssistantAgent
import tempfile

# Streamlit app
def main():
    st.set_page_config(page_title="Tax Provider", layout="wide")

    # Custom CSS to inject into the Streamlit app
    custom_css = """
    <style>
        .css-qrbaxs {background-color: #f0f2f6;}
        .stButton>button {background-color: #0d6efd; color: white;}
        .stTextInput>div>div>input {color: blue;}
        .stTextArea>div>div>textarea {background-color: #e9ecef;}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

    st.sidebar.title("Configuration")
    st.sidebar.markdown("## Provide Your Details Here")

    api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")
    os.environ['OPENAI_API_KEY'] = api_key

    uploaded_file = st.sidebar.file_uploader("Upload a text file", type=["txt"])

    # Initialize or update the conversation in session state
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []

    def add_to_conversation(participant, text):
        st.session_state.conversation.append((participant, text))

    def format_conversation():
        return "\n".join([f"{participant}: {text}" for participant, text in st.session_state.conversation])

    # Use Streamlit's session state to store the agent
    if 'global_agent' not in st.session_state:
        st.session_state.global_agent = None

    @st.cache_data()
    def initialize_agent(file_path):
        return OpenAIAssistantAgent.from_new(
             name="Betting Wizard",
            instructions="""The Serie A Predictor is an advanced tool for providing predictions on Serie A matches based on detailed data analysis. Here are the steps to follow:
            Analyze the Serie A Database (Mandatory Step):
            Check and analyze the Serie A database. This is an Excel file containing various worksheets with detailed league data, such as standings, team statistics, defensive phases, shots, passes, goal creation, defensive actions, ball possession, playing time, and more.
            Be sure to review all available information to get a complete and accurate view.
            Important: The database also includes detailed information and indicators on players. This data is critical for cross-referencing once formations are confirmed, allowing you to assess the impact of individual players on the game.
            Checking Confirmed Formations (Mandatory Step):
            Visit sport.sky.it to get the latest information on the confirmed lineups of Serie A teams. This step is crucial to understanding the teams' potential performances.
            Checking Live Odds (Mandatory Step):
            Consult the snai.it and sisal.it websites to check live odds for matches. The odds can provide valuable insights into market expectations.
            Important Points to Consider:
            The database contains several worksheets. It is essential to analyze all the data to cross-reference and get a complete view.
            Pay attention to the various categories of statistics, such as individual player statistics and miscellaneous statistics by team, to improve the accuracy of your predictions.
            Pay attention to various categories of statistics, including individual player statistics, to improve the accuracy of your predictions.
            Example System Response:
            1. UNDER 10.5 Total Corner Kicks.
            - Odds: 1.60
            -Estimated probability: 63%
            - Consideration: Based on the past trends of Juventus and Inter, this bet could be advantageous. Both teams tend to focus on positional play rather than creating many corner kick opportunities.""",
            openai_tools=[{"type": "retrieval"}],
            instructions_prefix="You are a professional Betting system  providing predictions on Serie A matches",
            files=[file_path],
            verbose=True,
        )

    if uploaded_file is not None and st.session_state.global_agent is None:
        # Save uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_file_path = tmp_file.name

        st.sidebar.success(f'Uploaded File: {uploaded_file.name}')
        st.session_state.global_agent = initialize_agent(temp_file_path)

    input_text = st.text_input("Input:")
    execute_button = st.button("Execute")

    if execute_button and st.session_state.global_agent:
        try:
            agent_response = st.session_state.global_agent.chat(input_text)
            response_text = agent_response.response if not isinstance(agent_response, str) else agent_response
            st.text_area("Output:", value=response_text, height=300)
            add_to_conversation("User", input_text)
            add_to_conversation("Agent", response_text)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    # Download button in the sidebar
    if st.sidebar.download_button(label="Download Conversation", data=format_conversation(), file_name="conversation.txt", mime="text/plain"):
        st.sidebar.success("Conversation downloaded!")

if __name__ == "__main__":
    main()
