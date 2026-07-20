import ollama
from search_tool import web_search

LLM="qwen3-vl:30b" 
TOOLS = {"web_search": web_search}

def model(history):
        response = ollama.chat( 
            model=LLM,
            messages=history,
            tools=list(TOOLS.values()),
            options={"temperature": 0}, 
        )
        msg =  response["message"]
        if msg.tool_calls:
            call = msg.tool_calls[0]
            return {"type": "tool", "name": call.function.name, "args": call.function.arguments,
                    "content": msg.content or "", "tool_calls": msg.tool_calls}

        return {"type": "final", "content": response["message"]["content"]}

SYSTEM_PROMPT = """
you have web_search tool that takes a query and returns web search results. 
use the web_search tool to get facts and sources to answer queries.
never answer search queries yourself. f the search results don't contain the answer, say so or search again. never fill the gap from your own knowledge.
Every fact in your final answer must cite the result number it came from, like [2]. If a claim has no source in the search results, do not make it.
If your own knowledge disagrees with a search result, trust the search result."
"""

def agent(task, model, tools, max_steps=10):
    history = [ 
             {"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": task}
             ]
    print(f"TASK: {task}\n")
    for step in range(1, max_steps+1):
        print(f"--- loop {step} ---")
        action = model(history)
        print(f"  1. model decided: {action}")

        if action["type"] == "final":
                print("  2. type is 'final' -> returning\n")
                return action["content"], history
        
        history.append({"role": "assistant", 
                                "content": action.get("content", ""),
                                "tool_calls": [action["tool_calls"][0]],})

        name = action["name"]
        args = action["args"]

        if name not in tools:
            result = f"unknown tool: {name}"
        else:
            try :
                result = tools[name](**args)    
            except Exception as e: 
                result = f"tool error in {name}: {type(e).__name__}: {e}"
        
        print(f"  3. ran tool {name}({args}) -> {result}")
        history.append({"role": "tool", 
                                "name": name,
                                "args":args, 
                                "content": str(result)})
        print(f"  4. appended result; history now has {len(history)} messages\n")
        print("history now is : " , history)
        print("\n\n")

    raise RuntimeError("agent did not finish — it looped")


if __name__ == "__main__":
    answer, _ = agent("Who influenced the artist who painted The Starry Night, and which museum shows one of those influences' works?", model, TOOLS)

    print(answer)
    