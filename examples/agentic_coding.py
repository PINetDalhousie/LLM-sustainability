#!/usr/bin/env python3
# Copyright (c) 2025 by Monzurul
# Multi-agents collaboration task: developing a snake game using vLLM
# This is done by a role-playing prompt method. The architecture of this workflow is:
#                       Architect
#  Coder (main.py) | Coder (snake.py) | Coder (game.py)
#                       Reviewer
#  Reviser (main.py) | Reviser (snake.py) | Reviser (game.py)

import time
from openai import OpenAI
import os
from zeus.monitor import ZeusMonitor
import argparse

# Initialize OpenAI client for vLLM server
client = OpenAI(
    api_key="token123",  
    base_url="http://localhost:8000/v1"
)

def parse_args():
    parser = argparse.ArgumentParser(description='Multi-agentic coding with configurable output token limit')
    parser.add_argument('--max_tokens', type=int, default=1000)

    return parser.parse_args()

def _make_completion(prompt, max_tokens=1000, temperature=0.7):
    """Make a completion request to vLLM server."""
    try:
        completion = client.completions.create(
            model="lmsys/vicuna-7b-v1.3",
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=None
        )
        response = completion.choices[0].text.strip()
        return response
    except Exception as e:
        print(f"Error making completion request: {e}")
        return ""

def architect(max_tokens=1200):
    """System architect that designs the structure and APIs."""
    prompt = """Role: You are a system architect.

User gives you a task. You design a list of files and design a list of APIs with full function signatures (with functionality as comments) for each file to achieve the task.

Task: Write a cli snake game in python.

Response in the format:

Files:
main.py
game.py
snake.py

APIs:
main.py:
Code:```
[API definitions for main.py with function signatures and comments]
```

game.py:
Code:```
[API definitions for game.py with function signatures and comments]
```

snake.py:
Code:```
[API definitions for snake.py with function signatures and comments]
```
"""
    print(f"max_tokens in architect: {max_tokens}")
    # response = _make_completion(prompt, max_tokens=1200)
    response = _make_completion(prompt, max_tokens=max_tokens)
    
    # Parse the response to extract APIs for each file
    apis = {"main_api": "", "game_api": "", "snake_api": ""}
    
    lines = response.split('\n')
    current_section = None
    current_code = []
    in_code_block = False
    
    for line in lines:
        if "main.py:" in line:
            current_section = "main_api"
            current_code = []
        elif "game.py:" in line:
            current_section = "game_api"
            current_code = []
        elif "snake.py:" in line:
            current_section = "snake_api"
            current_code = []
        elif line.strip().startswith("```") and current_section:
            if in_code_block:
                apis[current_section] = '\n'.join(current_code)
                in_code_block = False
            else:
                in_code_block = True
                current_code = []
        elif in_code_block and current_section:
            current_code.append(line)
    
    return apis["main_api"], apis["game_api"], apis["snake_api"]


def programmer(architect_response, file_name, other_filename1, other_filename2, max_tokens=800):
    """Expert programmer that implements the APIs."""
    prompt = f"""You are a Python programmer. Create a complete Snake game.

Task: Write the {file_name} file for a CLI Snake game.

Requirements:
- {file_name}: {"Main game loop and user interface" if file_name == "main.py" else "Snake class with movement logic" if file_name == "snake.py" else "Game class with board and food logic"}
- Use only standard Python libraries (no external dependencies)
- Make it a working command-line Snake game
- Include all necessary functions

Write ONLY Python code for {file_name}:

```python"""
    print(f"max_tokens in programmer: {max_tokens}")
    # response = _make_completion(prompt, max_tokens=800)
    response = _make_completion(prompt, max_tokens=max_tokens)
    
    # Debug: Print the raw response for troubleshooting
    print(f"\n=== DEBUG: {file_name} response (first 200 chars) ===")
    print(repr(response[:200]))
    print("=" * 50)
    
    # Extract code from response - look for python code blocks first
    if "```python" in response:
        code_parts = response.split("```python")
        if len(code_parts) >= 2:
            code = code_parts[1].split("```")[0].strip()
            return code
    elif "```" in response:
        code_parts = response.split("```")
        if len(code_parts) >= 3:  # Need at least opening, code, closing
            return code_parts[1].strip()
    
    # If no code blocks found, return the response as-is (it might be the code directly)
    return response.strip()


def reviewer(main_code, snake_code, game_code, max_tokens=500):
    """Expert code reviewer that provides feedback."""
    # Create a simple, short review to avoid token limits
    prompt = f"""Review this Python Snake game code and provide brief improvement suggestions:

Files created:
- main.py: {len(main_code)} characters
- snake.py: {len(snake_code)} characters  
- game.py: {len(game_code)} characters

Brief review comments:
- Focus on functionality and correctness
- Suggest improvements for Snake game logic
- Keep comments short and specific"""
    print(f"max_tokens in reviewer: {max_tokens}")
    # return _make_completion(prompt, max_tokens=500)
    return _make_completion(prompt, max_tokens=max_tokens)


def reviser(main_code, snake_code, game_code, file_name, review, max_tokens=500):
    """Code reviser that improves code based on review."""
    # Select only the code for the file being revised to keep prompt short
    if file_name == "main.py":
        current_code = main_code
    elif file_name == "snake.py":
        current_code = snake_code
    else:
        current_code = game_code
    
    prompt = f"""You are a Python programmer. Improve this Snake game code:

Current {file_name}:
{current_code}

Write ONLY the improved Python code for {file_name}. Make it functional:

```python"""
    print(f"max_tokens in reviser: {max_tokens}")
    # response = _make_completion(prompt, max_tokens=500)
    response = _make_completion(prompt, max_tokens=max_tokens)
    
    # Debug: Print the raw response for troubleshooting
    print(f"\n=== DEBUG: {file_name} reviser response (first 200 chars) ===")
    print(repr(response[:200]))
    print("=" * 50)
    
    # Extract code from response - the model should continue from "```python"
    # So the response should be the Python code directly
    if response.strip():
        # Remove any trailing ``` if present
        code = response.split("```")[0].strip()
        return code
    
    return ""


def main():
    print("Starting Multi-Agent Snake Game Development with vLLM...")

    args = parse_args() 
    max_tokens=args.max_tokens
    print(f"Max output tokens set to:{max_tokens}")
    
    # Start Zeus monitoring for the entire workflow
    zeus_monitor = ZeusMonitor()
    start_time = time.time()
    zeus_monitor.begin_window("multi_agent_coding_workflow")
    
    print("Phase 1: Architecture Design...")
    main_api, game_api, snake_api = architect(max_tokens)
    
    print("Phase 2: Initial Coding...")
    code_main = programmer(main_api, "main.py", "game.py", "snake.py", max_tokens)
    code_snake = programmer(snake_api, "snake.py", "main.py", "game.py", max_tokens)
    code_game = programmer(game_api, "game.py", "main.py", "snake.py", max_tokens)
    
    print("Phase 3: Code Review...")
    review = reviewer(code_main, code_snake, code_game, max_tokens)
    
    print("Phase 4: Code Revision...")
    revised_code_main = reviser(code_main, code_snake, code_game, "main.py", review, max_tokens)
    revised_code_snake = reviser(code_main, code_snake, code_game, "snake.py", review, max_tokens)
    revised_code_game = reviser(code_main, code_snake, code_game, "game.py", review, max_tokens)
    
    # End Zeus monitoring
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nMulti-agent workflow completed in {total_time:.2f} seconds")
    
    mes = zeus_monitor.end_window("multi_agent_coding_workflow")
    # Calculate totals
    total_gpu = sum(mes.gpu_energy.values()) if mes.gpu_energy else 0
    total_cpu = sum(mes.cpu_energy.values()) if mes.cpu_energy else 0
    total_dram = sum(mes.dram_energy.values()) if mes.dram_energy else 0
    total_energy = total_gpu + total_cpu + total_dram
    
    print(f"Application consumed Time: {mes.time} s, CPU energy: {total_cpu} J, GPU energy: {total_gpu} J, DRAM energy: {total_dram} J & Total energy: {total_energy} J during inference.")
    print(f"Energy (Zeus): {mes}")
    
    # Print results
    print("\n" + "=" * 60)
    print("FINAL CODE OUTPUTS")
    print("=" * 60)
    
    print("\n---------- main.py ----------")
    print(revised_code_main)

    print("\n---------- snake.py ----------")
    print(revised_code_snake)

    print("\n---------- game.py ----------")
    print(revised_code_game)
    
    print("\n" + "=" * 60)
    print("CODE GENERATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
