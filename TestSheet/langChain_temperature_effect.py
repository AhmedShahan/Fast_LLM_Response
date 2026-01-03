'''
Basic funda is
langchain function. 

same prompt

diferent different temerature
'''
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

prompt="Whst is AI"

import numpy as np
import time
temperature = np.arange(0.2, 0.91, 0.1)  # stop slightly above 0.9 to include 0.9
# print(temperature)
from chatbot.langchain_function import langchainChatbot

response_time=[]
max_token=200
for temp in temperature:
    print(f"Iteration for {temp}")
    print("*"*50)
    print("Lang Chain Start")
    stime=time.time()
    langchain_response=langchainChatbot(temperature=temp, max_output_tokens=max_token, query=prompt)
    print("Langchain response done .............")
    etime=time.time()
    response_time.append((etime-stime))
    

print(response_time)
'''
is there any effect on temperature inclreased or decreased on response time???

5:20 PM
No, temperature has NO significant effect on response time!

Why Temperature Doesn't Affect Speed:
Temperature শুধু output quality/randomness control করে, speed না।

What Temperature Actually Does:
python
temperature = 0.0  # Deterministic, same answer every time
temperature = 0.7  # Balanced creativity
temperature = 1.5  # Very creative/random
Low temperature (0.0-0.3): Model সবচেয়ে probable token select করে
High temperature (0.7-2.0): Model আরও random/creative token select করে
কিন্তু token generation process একই থাকে!

What ACTUALLY Affects Response Time:
1. max_output_tokens ← This is THE KEY! ⚡
python
# Fast (2-3 seconds)
max_output_tokens = 50

# Medium (5-10 seconds)  
max_output_tokens = 500

# Slow (15-30 seconds)
max_output_tokens = 2000
কারণ: LLM একটা একটা করে token generate করে। বেশি token = বেশি সময়।

2. Prompt Complexity
python
# Fast
"Hi"

# Medium
"Explain AI"

# Slow  
"Write a detailed 2000-word essay on the history of AI with examples"
3. Model Size
python
# Fastest
model = "gemini-2.0-flash-exp"

# Medium
model = "gemini-2.5-flash"  

# Slowest (but most capable)
model = "gemini-2.0-pro"
Quick Experiment You Can Try:
python
import time

prompts = ["What is AI"] * 3
temperatures = [0.0, 0.7, 1.5]
max_tokens = 200

for temp in temperatures:
    start = time.time()
    response = langchainChatbot(temp, max_tokens, "What is AI")
    elapsed = time.time() - start
    print(f"Temperature {temp}: {elapsed:.2f}s")
```

**Expected Result:**
```
Temperature 0.0: 3.45s
Temperature 0.7: 3.52s  ← Almost same!
Temperature 1.5: 3.48s
যদি Response Time কমাতে চান:
✅ এগুলো কাজ করবে:
python
# 1. Reduce max_output_tokens (BIGGEST IMPACT!)
max_output_tokens = 100  # instead of 500

# 2. Use faster model
model = "gemini-2.0-flash-exp"

# 3. Use streaming (perceived speed)
for chunk in llm.stream(query):
    print(chunk.content, end="")

# 4. Shorter, clearer prompts
query = "Define AI briefly"  # instead of long prompts
❌ এগুলো কাজ করবে না:
python
# Changing temperature (no effect on speed)
temperature = 0.1  # ❌ Won't make it faster

# Changing top_p, top_k (minimal effect)
top_p = 0.9  # ❌ Won't make it faster
Real Performance Test:
python
import time

def test_speed_factors():
    # Test 1: Temperature effect (should be ~same)
    print("=== Temperature Test ===")
    for temp in [0.0, 0.5, 1.0, 1.5]:
        start = time.time()
        langchainChatbot(temp, 200, "What is AI")
        print(f"Temp {temp}: {time.time()-start:.2f}s")
    
    print("\n=== Max Tokens Test ===")
    # Test 2: Max tokens effect (should vary a lot!)
    for tokens in [50, 200, 500, 1000]:
        start = time.time()
        langchainChatbot(0.7, tokens, "Explain AI in detail")
        print(f"Tokens {tokens}: {time.time()-start:.2f}s")

test_speed_factors()
```

**Expected Output:**
```
=== Temperature Test ===
Temp 0.0: 3.2s    ← Similar
Temp 0.5: 3.4s    ← Similar  
Temp 1.0: 3.1s    ← Similar
Temp 1.5: 3.3s    ← Similar

=== Max Tokens Test ===
Tokens 50: 2.1s      ← Fast!
Tokens 200: 5.3s     ← Medium
Tokens 500: 12.8s    ← Slow
Tokens 1000: 24.5s   ← Very slow
Summary:
Parameter	Effect on Speed	Effect on Quality
temperature	❌ No effect	✅ Yes (creativity)
max_output_tokens	✅ Big effect	✅ Yes (length)
Model choice	✅ Big effect	✅ Yes (capability)
Prompt length	✅ Small effect	✅ Yes (context)
Bottom line: আপনি যদি fast response চান, তাহলে max_output_tokens কমান, temperature change করে লাভ নেই! 🚀
'''