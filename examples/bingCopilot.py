#!/usr/bin/env python3
# Copyright (c) 2025 by Monzurul
# Author: Monzurul

# This application simulates a Bing Copilot-like experience with:
# - Large system prompt (~1200 tokens)
# - User query processing
# - Response generation limited to 800 tokens
# - Uses vLLM server via OpenAI-compatible API

import time
import tiktoken
import sys
import os
from openai import OpenAI
from zeus.monitor import ZeusMonitor

# Add the current directory to the Python path to allow importing the system_prompts module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from system_prompts.bing_copilot_prompt_1200_token import generate_system_prompt

# Initialize the tokenizer for token counting
tokenizer = tiktoken.get_encoding("cl100k_base")  # Using OpenAI's tokenizer as an approximation

def count_tokens(text):
    """Count the number of tokens in a text string."""
    return len(tokenizer.encode(text))

# Initialize OpenAI client for vLLM server
client = OpenAI(
    api_key="token123",  # vLLM doesn't validate this
    base_url="http://localhost:8000/v1"
)


def bing_copilot(system_prompt, user_query, max_tokens=800):
    """Generate a response using the vLLM server via OpenAI-compatible API."""
    # Construct the full prompt
    full_prompt = f"""{system_prompt}

User Query: {user_query}

Response (max {max_tokens} tokens):"""
    
    # Call the vLLM server using OpenAI completions API
    completion = client.completions.create(
        model="lmsys/vicuna-7b-v1.3",  # This should match the model running on vLLM server
        prompt=full_prompt,
        max_tokens=max_tokens,
        temperature=0.7,
        top_p=1.0,
        echo=False
    )
    
    return completion.choices[0].text.strip()


def process_query(system_prompt, user_query):
    """Process a single query and display the results."""
    print("\nProcessing your query...")
    start_time = time.time()
    
    # Initialize Zeus monitor for energy measurement
    zeus_monitor = ZeusMonitor()
    zeus_monitor.begin_window("LLM_online_inference")
    
    # Call the bing_copilot function with the system prompt and user query
    response = bing_copilot(system_prompt, user_query)
    
    # End energy monitoring
    mes = zeus_monitor.end_window("LLM_online_inference")
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print("\n" + "="*50)
    print(f"Response (generated in {processing_time:.2f} seconds):")
    print("="*50)
    print(response)
    
    # Display token counts
    response_tokens = count_tokens(response)
    # Check token count of system prompt
    token_count = count_tokens(system_prompt)
    print("\n" + "-"*50)
    print(f"System prompt: {token_count} tokens | Response: {response_tokens} tokens")
    print(f"Energy consumed: {mes.time:.2f}s, {mes.total_energy:.2f}J")
    print("-"*50)
    
    return response, processing_time, response_tokens


def main():
    """Main function to run the Bing Copilot application."""
    print("Initializing Bing Copilot...")
    
    # Generate the system prompt
    system_prompt = generate_system_prompt()
    # print(system_prompt
    
    # Check if a query was provided as a command-line argument
    if len(sys.argv) > 1:
        # Non-interactive mode with query from command line
        user_query = sys.argv[1]
        process_query(system_prompt, user_query)
    else:
        # Interactive mode
        print("\n" + "="*50)
        print("Bing Copilot Initialized")
        print("="*50)
        
        while True:
            # Get user query
            user_query = input("\nEnter your query (or 'exit' to quit): ")
            if user_query.lower() == 'exit':
                break
                
            process_query(system_prompt, user_query)


# Run the application
if __name__ == "__main__":
    main()
