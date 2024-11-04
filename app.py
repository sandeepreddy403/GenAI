#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
from walmart_multi_agent_system import run_multi_agent_system

st.title("Multi-Agent System for GenAI Use Cases")

# Create an input field
user_input = st.text_input("Enter input for the multi-agent system:")

# Process input and display output when user submits
if st.button("Run Multi-Agent System"):
    result = run_multi_agent_system(user_input)
    st.write("Result:", result)

