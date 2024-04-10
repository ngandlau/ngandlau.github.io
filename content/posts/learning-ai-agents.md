---
title: "Learning Ai Agents"
date: 2024-04-10T19:28:53+02:00
draft: false
---

📚 https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/ 

An OpenAI researcher that summarizes well the delicacies about prompt engineering.
For example, ending a prompt with `\n` likely outperforms one that ends with `.`. 
Or a prompt with `Question: ...` likely outperforms one with `Q: ...`.

The tricks will be different for each model. For example, using `Q: ...` might be favored when using Anthropic, and `Question: ...` when using OpenAI.
I think there is a need for *prompt optimizers* that take in a prompt, and optimize it for a given model.
The prompt optimizer knows all the tricks for each language models.
This should decrease the amount of trial & error (and thus cost!) when developing AI systems.

Note that when you play with prompts, two things are worth knowing:

1. What changes in a prompt are *not* worth trying? For example, experiments may have shown that changing the system prompt from `You are a helpful AI system.` to `You are a helpful agent.` is likely to not alter performance. 
2. What changes in a prompt are *worth* trying?

--- 

📚 Here is one of fundamental "system" prompts of `gpt-researcher`:

```
"You are an AI critical thinker research assistant. Your sole purpose is to write well written, critically acclaimed, objective and structured reports on given text."
```

--- 

A common **ReAct** prompt template:

```python
react_prompt_template = """You are a thoughtful assistant. Answer the following questions as best you can. You have access to the following tools:

{tools}
<function_name>: <function_description> <function_arg_description_and_types>
<function_name>: <function_description> <function_arg_description_and_types>
<function_name>: <function_description> <function_arg_description_and_types>

You must use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}
"""
```

---

Despite prompting the LLM using a ReAct template, I cannot get him to use the format `question: ... thought: ... action: ... ` when calling it with OpenAI functions/tools. Here's what I get. It seems like OpenAI functions/tools should be used as a separate agent that only selects & executes the right function. It should not be the same agent that also does the planning.

![[Pasted image 20240408212851.png]]

When the `tools`  argument is commented out, the llm responds with the prompted question-thought-action format:

![[Pasted image 20240408213329.png]]
![[Pasted image 20240408213407.png]]

So, internally OpenAI has defined its own prompt that determines how the LLM (1) chooses and (2) uses a function.
Consequently, I should split my agent into (1) a planner agent and (2) a choose-and-execute-function-agent.

---

📚 https://arxiv.org/html/2401.02777v1 & https://arxiv.org/pdf/2401.02777v1.pdf
The proposed Prompting Technique **RAISE** seems to be a framework that *improves upon ReAct* (Reason + Act).

![[Pasted image 20240410103548.png]]

In the image below, it looks like the `agent_scratchpad` contains the observation or results of the previous "run".
Or probably all previous observations.

![[Pasted image 20240410104819.png]]

--- 

💭 Some papers & blogs talk about **task solving trajectories**. What are they?

These are tuples of (thought, act) that are generated at each "run" or "step" the ReAct-Agent makes (https://www.promptingguide.ai/techniques/react).
I think they can be passed to the agent's `agent_scratchpad` at step `t+1` to inform the agent about the progress he has made so far.

Later in the [guide](https://www.promptingguide.ai/techniques/react), the author says that task-solving trajectories are (thought, action, observation) pairs.
I conclude that a *task solving trajectory* is not clearly defined and simply is a summary of what the LLM has done so far.
It's like a crossed-out todo list, but with more infos such as the reasoning that lead to the creation of the task (thought).
The summary should be optimized to be consumed by an LLM.

---

**Observation**
> "Obs corresponds to observation from the environment that's being interacted with (e.g., Search engine)."

I think it's *not* the raw output of a tool, e.g. the raw text of a web search tool, but the LLMs summary of that raw text.
If that is true, then the observation can be thought of as a function of the raw tool output: `observation = f(tool_output)`.

💭 different prompts are used for different types of tasks. (source: https://www.promptingguide.ai/techniques/react)
- for tasks where *reasoning* is of primary importance, multiple *thought-action-observation* steps are used for the *task solving trajectory*.
- for tasks where *decision-making* is of primary importance, the agent will use lots of *action steps*, and *thoughts* are of lesser importance.

A potential problem: 
> "ReAct depends a lot on the information it's retrieving; non-informative search results derails the model reasoning and leads to difficulty in recovering and reformulating thoughts"

---

▶️ https://blog.langchain.dev/automating-web-research/ & https://github.com/langchain-ai/web-explorer/blob/main/web_explorer.py
They came up with the procedure that everyone would come up with, but with a slight twist at the end:

1. use an LLM to generate `N` search queries
2. execute 1 search query (using a Bing/Google/Taviliy search tool)
3. choose top `K` links per query
4. scrape information from all `N*K` links
5. store the scraped information (documents) in a *vectorscore*
6. find the most relevant documents for each original search query

So the thing that they do differently compared to `gpt-researcher` is that they store the scraped websites in a *vectorstore* and then run a similarity search (vector similarity? vector cosine?) against the original search query.

---

📚 https://python.langchain.com/docs/modules/agents/how_to/custom_agent/
Here's a hint of how the `agent_scratchpad` actually looks like:

```python
{"agent_scratchpad": lambda x: format_to_openai_tool_messages(x["intermediate_steps"])}
```

--- 

💡 **SalesGPT**, a template built using langchain, utilizes the concept that the creator of langchain talks about in a recent podcast: *"agents as state machines"*.

💭 The idea is to create a prompt that defines all states in which a conversation can be in. 

Consider a conversation that occurs when a *seller* reaches out to a potential *customer*. 
At first, the seller should introduce himself. This is stage `"1": "Introduction`.
At some point, the seller should reveal his intentions and describe the product he wants to sell. This is stage `"3": "Value proposition"`.
And so forth.


```
conversation_stage_dict: Dict = {
	"1": "Introduction: Start the conversation by introducing yourself and your company. Be polite and respectful while keeping the tone of the conversation professional. Your greeting should be welcoming. Always clarify in your greeting the reason why you are contacting the prospect.",
	"2": "Qualification: Qualify the prospect by confirming if they are the right person to talk to regarding your product/service. Ensure that they have the authority to make purchasing decisions.",
	"3": "Value proposition: Briefly explain how your product/service can benefit the prospect. Focus on the unique selling points and value proposition of your product/service that sets it apart from competitors.",
	"4": "Needs analysis: Ask open-ended questions to uncover the prospect's needs and pain points. Listen carefully to their responses and take notes.",
	"5": "Solution presentation: Based on the prospect's needs, present your product/service as the solution that can address their pain points.",
	"6": "Objection handling: Address any objections that the prospect may have regarding your product/service. Be prepared to provide evidence or testimonials to support your claims.",
	"7": "Close: Ask for the sale by proposing a next step. This could be a demo, a trial or a meeting with decision-makers. Ensure to summarize what has been discussed and reiterate the benefits.",
}
```

You can consider each state as a *node*. 
When you are in a node you can transition to other nodes. 
Transitions from one node to another node could be described using *transition probabilties*.
Probably you can design the prompt such that the LLM decides on his own which node to transition to next. 
In that case, transition probabilities are implicit.
But you could probably design the prompt such that the LLM is nudged to prefer certain transitions. 
In that case, transition probabilities are both implicit (LLM still reasons on his own) and explicit (in your prompt, you tell the LLM which transitions to prefer).

What I don't like is that LangChain is also using the graph datastructures (nodes, edges, transition probabilities) for designing multi-agent systems.
In their mental model for multi-agent systems, each agent is a node. And agent `A` can deligate work to agent `B`. For example, a `planner agent` might call a `research agent` to do research.
