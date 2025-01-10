import streamlit as st
from openai import OpenAI

# Set up the OpenAI API key
client = OpenAI(api_key="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")

# Streamlit app title
st.title("Enhanced Dashboard with ChatGPT Integration")

# Sidebar for ChatGPT input
st.sidebar.header("ChatGPT Assistant")
user_input = st.sidebar.text_area("Ask ChatGPT:", placeholder="Type your question here...")

if st.sidebar.button("Send"):
    if user_input:
       completion = client.chat.completions.create(
           model="gpt-3.5-turbo",
           messages={"role": "system", "content": f"{user_input}"}
       )
       print(completion.choices[0].message)


