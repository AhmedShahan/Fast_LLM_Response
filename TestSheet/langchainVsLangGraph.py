import sys
import os

# Add the project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
'''
Basic Motive is to check 
Langchain response time and langgraph response

same prompt but there 5 prompt one bu one
this should be sequencial not parallel execution

same llm configuration
'''

promps=[
    "Hi",
    "How are you today?",
    "What is AI",
    "Who is the father of AI",
    "What is the difference between AI and ML"
]



### llm configuration
temperature=0.9
max_token=200

from chatbot.langchain_function import langchainChatbot
from chatbot.langgraph_function import langGraphChatbot

LC_response_time=[]
LG_response_time=[]

import time
for prompt in promps:
    print("Lang Chain Start")
    stime=time.time()
    langchain_response=langchainChatbot(temperature=temperature, max_output_tokens=max_token, query=prompt)
    print("Langchain response done .............")
    etime=time.time()
    LC_response_time.append((etime-stime))
    print("Lang Graph Start")
    stime=time.time()
    langGraph_response=langGraphChatbot(temperature=temperature, max_output_tokens=max_token, query=prompt)
    etime=time.time()
    LG_response_time.append((etime-stime))
    print("Graph response done .............")


print("LangChain Time: ", LC_response_time)
print("LangGraph Time: ", LG_response_time)



'''
LangChain Time:  [2.7755322456359863, 4.624027490615845, 11.671765804290771, 46.04394769668579, 10.827744960784912]
LangGraph Time:  [4.013000249862671, 4.660099506378174, 11.164138793945312, 6.539490699768066, 12.58867883682251]



Main Reasons for Time Variations
1. Network Latency

Each API call goes through the internet to Google's servers
Network conditions vary: congestion, routing, packet loss
Distance to the nearest Google API server can fluctuate

2. API Server Load

Google's servers handle millions of requests
Server load varies based on:

Time of day
Other users' concurrent requests
Server availability and distribution



3. LLM Processing Complexity

Different prompts require different processing:

"Hi" → Very simple, quick response
"What is the difference between AI and ML" → More complex reasoning, longer generation time


Token generation is sequential, so longer responses take more time

4. Cold Start vs Warm Requests

First request might initialize connections
Subsequent requests may benefit from cached connections
But this can vary unpredictably

5. LangGraph Overhead

LangGraph adds a graph compilation and execution layer
This overhead is relatively consistent but adds to total time
Sometimes this overhead is offset by other factors being faster

Your Specific Results Analysis
Prompt 4: "Who is the father of AI"
- LangChain: 46.04s (unusually slow!)
- LangGraph: 6.54s (normal)
This massive spike (46 seconds) suggests:

Network timeout/retry
Server-side queueing
Temporary API slowdown

How to Get More Consistent Results
pythonimport time
import statistics

def benchmark_with_multiple_runs(prompt, runs=3):
    """Run same prompt multiple times and get statistics"""
    lc_times = []
    lg_times = []
    
    for i in range(runs):
        # LangChain
        start = time.time()
        langchainChatbot(temperature=0.9, max_output_tokens=200, query=prompt)
        lc_times.append(time.time() - start)
        
        time.sleep(1)  # Small delay between calls
        
        # LangGraph
        start = time.time()
        langGraphChatbot(temperature=0.9, max_output_tokens=200, query=prompt)
        lg_times.append(time.time() - start)
        
        time.sleep(1)
    
    return {
        'lc_avg': statistics.mean(lc_times),
        'lc_median': statistics.median(lc_times),
        'lc_stdev': statistics.stdev(lc_times) if len(lc_times) > 1 else 0,
        'lg_avg': statistics.mean(lg_times),
        'lg_median': statistics.median(lg_times),
        'lg_stdev': statistics.stdev(lg_times) if len(lg_times) > 1 else 0,
    }

# Test with multiple runs
results = benchmark_with_multiple_runs("What is AI", runs=5)
print(f"LangChain - Avg: {results['lc_avg']:.2f}s, Median: {results['lc_median']:.2f}s")
print(f"LangGraph - Avg: {results['lg_avg']:.2f}s, Median: {results['lg_median']:.2f}s")
Expected Performance Pattern
For a simple chatbot like yours:

LangChain: Should be slightly faster (less overhead)
LangGraph: Adds ~0.5-2s overhead for graph compilation/execution



######## ###########################################################
তার মানে আমার থেকে fast করার কোনও কিছু নেই। মানে এই মেথডে আমি LangChain LangGraph যাই ইয়হার কওরই almost same response আসবে??4:57 PMহ্যাঁ, একদম ঠিক বলেছেন!
আপনার Current Use Case এ:
আপনি যেহেতু শুধু simple single-turn chatbot বানাচ্ছেন (একটা prompt → একটা response), সেখানে:

✅ LangChain এবং LangGraph প্রায় same performance দেবে
✅ আসল সময় লাগছে Google API call এ (network + LLM processing)
✅ Framework এর overhead মাত্র 0.5-2 seconds, যা negligible

কখন LangGraph দরকার?
LangGraph তখনই দরকার যখন আপনার complex workflow আছে:
❌ দরকার নেই (আপনার case):
python# Simple: User → LLM → Response
user_input → llm.invoke() → response
✅ দরকার আছে:
python# Complex workflow উদাহরণ:

1. Multi-step reasoning:
   user_input → research_node → analyze_node → summarize_node → response

2. Conditional logic:
   user_input → classify_node 
              ├→ technical_question → code_node
              └→ general_question → chat_node

3. Agent with tools:
   user_input → agent_node → [web_search, calculator, database]
              → combine_results → response

4. Human-in-the-loop:
   user_input → draft_node → wait_for_approval → finalize_node
আপনার Code এর জন্য Recommendation:
শুধু LangChain ব্যবহার করুন! আপনার প্রথম code টাই যথেষ্ট:
pythondef langchainChatbot(temperature, max_output_tokens, query) -> str: 
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    response = llm.invoke(query)
    return response.content
এটাই fastest এবং simplest আপনার use case এর জন্য।
Response Time Fast করার উপায়:
যদি আপনি সত্যিই faster response চান:
1. Streaming ব্যবহার করুন:
pythondef langchainChatbot_streaming(temperature, max_output_tokens, query):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    
    # User কে real-time এ response দেখান (ChatGPT এর মতো)
    for chunk in llm.stream(query):
        print(chunk.content, end="", flush=True)
এতে user মনে করবে response faster আসছে!
2. Faster Model ব্যবহার করুন:
pythonmodel="gemini-2.0-flash-exp"  # আরও fast
3. Caching (যদি same question বারবার আসে):
pythoncache = {}

def langchainChatbot_cached(temperature, max_output_tokens, query):
    if query in cache:
        return cache[query]  # Instant response!
    
    llm = ChatGoogleGenerativeAI(...)
    response = llm.invoke(query)
    cache[query] = response.content
    return response.content
Summary:

🎯 আপনার current use case এ: LangChain যথেষ্ট, LangGraph দরকার নেই
⚡ Fast করতে চাইলে: Streaming ব্যবহার করুন
🔄 Complex workflow থাকলে: তখন LangGraph এ যাবেন

আশা করি clear হয়েছে! আরও কিছু জানতে চাইলে বলবেন। 😊
'''