# This python script demonstrates how to use LangChain to summarize a document in chunks.
# It uses the VLLM model from the langchain_community library to perform the summarization.
# The document is split into chunks of a specified size, and each chunk is summarized.
# The first chunk is summarized using a first prompt, and subsequent chunks are refined based on the previous summary.
# The final summary is printed at the end.
# Author: Monzurul
# Date: 2025-06-09
from langchain_openai import OpenAI
from langchain import LLMChain, PromptTemplate
from zeus.monitor import ZeusMonitor


llm = OpenAI(
    openai_api_key="token123",
    openai_api_base="http://localhost:8000/v1",
    model_name="lmsys/vicuna-7b-v1.3"         #"facebook/opt-125m"
)

first_prompt = PromptTemplate(
    input_variables=["doc", "word_limit"],
    template="""
The following is a piece of a document:
{doc}
Based on this piece of docs, summarize the main content as short as possible (≤ {word_limit} words).
Summary:
"""
)

refine_prompt = PromptTemplate(
    input_variables=["new_text", "previous_sum", "word_limit"],
    template="""
We have an existing summary:
{previous_sum}

New document segment:
{new_text}

If this adds new useful info, refine the summary (≤ {word_limit} words); else keep it unchanged.
Refined Summary:
"""
)

first_chain = LLMChain(llm=llm, prompt=first_prompt)
refine_chain = LLMChain(llm=llm, prompt=refine_prompt)

def chain_summarize(docs_path, chunk_size=2000, word_limit=100):
    # segments = split_into_chunks(text, size=chunk_size)
    docs = open(docs_path, "r").read().split("\n\n")
    summary = None

    cur_chunk = ""
    i = 0
    for chunk in docs:
        cur_chunk += chunk
        if len(cur_chunk) > chunk_size:
            print("Created chunk of size", len(cur_chunk))
            if i == 0:
                summary = first_chain.run(doc=cur_chunk, word_limit=word_limit)
            else:
                summary = refine_chain.run(
                    new_text=cur_chunk,
                    previous_sum=summary,
                    word_limit=word_limit,
                )
            
            i += 1
            cur_chunk = ""

    print("Total number of chunks:", i)
    print("\nFinal summary:\n", summary)
    return summary


docs_path = "data/state_of_the_union.txt"

zeus_monitor = ZeusMonitor()
zeus_monitor.begin_window("LLM_online_inference")
summary = chain_summarize(docs_path, chunk_size=2000, word_limit=100)
mes = zeus_monitor.end_window("LLM_online_inference")
print(f"Online inference consumed {mes.time} s and {mes.total_energy} J during inference.")
print(f"Energy (Zeus): {mes}")