#!/usr/bin/env python
# coding: utf-8

# multi-agent system using LangChain integrating with Hugging face mode for local inference.


# Import necessary libraries
import streamlit as st
import pandas as pd
from langchain.llms import HuggingFacePipeline
from bs4 import BeautifulSoup
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, Tool
from transformers import pipeline
import requests


# In[66]:


# Function to fetch Hugging Face links
def fetch_huggingface_links(search_query):
    """Fetch Hugging Face model links based on search query."""
    url = f"https://huggingface.co/models={search_query.replace(' ', '+')}"
    response = requests.get(url)
    if response.status_code != 200:
        st.write("Failed to retrieve data from Hugging Face.")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    model_links = []
    
    # Identify model links on Hugging Face search page
    for link in soup.find_all('a', href=True):
        if "/models/" in link['href']:
            full_link = "https://huggingface.co" + link['href']
            if full_link not in model_links:
                model_links.append(full_link)
            if len(model_links) >= 5:
                break

    if not model_links:
        st.write(f"No Hugging Face links found for '{search_query}'.")
    return model_links


# In[67]:


# Function to fetch GitHub links
def fetch_github_links(search_query):
    """Fetch GitHub repository links based on search query."""
    url = f"https://github.com/search={search_query.replace(' ', '+')}"
    response = requests.get(url)
    if response.status_code != 200:
        st.write("Failed to retrieve data from GitHub.")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    repo_links = []
    
    # Identify repository links on GitHub search page
    for link in soup.find_all('a', href=True):
        if "github.com" in link['href'] and "/blob/" not in link['href']:
            full_link = link['href']
            if full_link.startswith("https://github.com/") and full_link not in repo_links:
                repo_links.append(full_link)
            if len(repo_links) >= 5:
                break
                
    if not repo_links:
        st.write(f"No GitHub links found for '{search_query}'.")
    return repo_links


# # Define Agents

# # Industry Research Agent

# In[68]:


def industry_research_agent(company_name):
    """Research the industry and company's offerings."""
    industry_info = "The retail industry is highly competitive, with a significant focus on e-commerce, supply chain optimization, and customer experience enhancement."
    company_info = {
        "Name": company_name,
        "Key Offerings": "Groceries, electronics, home goods, and various other consumer products.",
        "Strategic Focus Areas": "Efficient supply chain management, competitive pricing, and enhancing customer experience."
    }
    st.write("Industry Information:", industry_info)
    st.write(f"{company_name}'s Offerings and Strategic Focus:", company_info)
    
    return industry_info, company_info


# # Market standards and use case genaration agent

# In[69]:


def market_standards_use_case_generation(industry_info, company_info, llm):
    """Generates AI and GenAI use cases based on industry standards."""
    prompt = PromptTemplate(
        input_variables=["industry_info", "company_info"],
        template=(f"In the retail industry, specifically for {company_info['Name']}, "
                  f"with offerings in {company_info['Key Offerings']} and a focus on {company_info['Strategic Focus Areas']}, "
                  "suggest relevant AI and GenAI use cases to optimize inventory, enhance customer satisfaction, "
                  "and improve operational efficiency using cutting-edge technologies.")
    )
    chain = LLMChain(prompt=prompt, llm=llm)
    use_cases = chain.run(industry_info=industry_info, company_info=company_info)
    st.write("Generated Use Cases:", use_cases)
    return use_cases


# #  Resource Asset Collection Agent

# In[70]:


def resource_asset_collection_agent(use_cases):
    """Collect resources for datasets and models based on use cases."""
    search_terms = [
        "retail demand forecasting AI",
        "customer recommendation system",
        "inventory optimization AI",
        "sentiment analysis for retail",
        "retail chatbot AI"
    ]
    
    all_links = {}
    
    for term in search_terms:
        # Collecting resources from Hugging Face and GitHub
        hf_links = fetch_huggingface_links(term)
        gh_links = fetch_github_links(term)
        all_links[term] = {
            "Hugging Face": hf_links,
            "GitHub": gh_links
        }
        st.write(f"Links for '{term}':")
        st.write("Hugging Face:", hf_links)
        st.write("GitHub:", gh_links)
    
    # Formatting links for the final report
    resource_links = "\n".join([f"- {term}:\n  Hugging Face: {', '.join(all_links[term]['Hugging Face'])}\n  GitHub: {', '.join(all_links[term]['GitHub'])}" for term in search_terms])
    st.write("Resource Asset Collection:", resource_links)
    return resource_links


# # Report compilation agent

# In[71]:


def report_compilation_agent(company_name, industry_info, company_info, use_cases, resources):
    """Compile the final proposal report."""
    report = f"""
    {company_name} AI and GenAI Use Case Proposal
    -----------------------------
    1. Industry Overview:
       {industry_info}
    
    2. {company_name}'s Offerings and Strategic Focus:
       - Key Offerings: {company_info['Key Offerings']}
       - Strategic Focus Areas: {company_info['Strategic Focus Areas']}
    
    3. Suggested Use Cases:
       {use_cases}
    
    4. Resource Asset Links:
       {resources}
    """
    st.write(report)
    
    # Optionally save to a file
    with open("use_case_proposal.txt", "w") as file:
        file.write(report)
    st.write("Final proposal saved as use_case_proposal.txt")
    return report


# # Main Function to Run the Multi-Agent System

# In[72]:


def main(company_name):
    # Initialize the LLM (Assuming using HuggingFace GPT-Neo)
    generator = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B", max_new_tokens=100)
    llm = HuggingFacePipeline(pipeline=generator)

    # Step 1: Research the industry and company's offerings
    industry_info, company_info = industry_research_agent(company_name)

    # Step 2: Generate use cases based on industry standards for the company
    use_cases = market_standards_use_case_generation(industry_info, company_info, llm)

    # Step 3: Collect resource assets based on use cases
    resources = resource_asset_collection_agent(use_cases)

    # Step 4: Compile the final report
    report = report_compilation_agent(company_name, industry_info, company_info, use_cases, resources)

    return report


# In[73]:


# Execute the multi-agent system
final_report = main("Walmart")

def run_multi_agent_system(user_input):
    # Your multi-agent system code logic here
    return "Output based on user input"



