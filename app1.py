import streamlit as st

# Title of the app
st.title("My First Streamlit App")

# Write some text
st.write("Welcome to my first Streamlit app!")

# Add a slider
number = st.slider("Pick a number", 0, 10)

# Display the result
st.write(f"You selected: {number}")