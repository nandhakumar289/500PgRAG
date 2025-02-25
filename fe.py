import streamlit as st

st.set_page_config(page_title="Sarah", layout="centered", page_icon="⚡")

from rag import orchestrator


st.header("Ask me anything!")
# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("What's on your mind?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👨"):
        st.write(prompt)

    # Add assistant response
    with st.chat_message("assistant", avatar="🤖"):

        response = f"Sarah: {prompt}"
        llm_response = orchestrator(prompt)
        st.write(llm_response)
        st.session_state.messages.append({"role": "assistant", "content": llm_response})
