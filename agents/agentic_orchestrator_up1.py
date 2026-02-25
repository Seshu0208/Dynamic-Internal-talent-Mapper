from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from agents.matcher_agent import score_match
from agents.skill_extraction_agent import extract_structured_skills
from tools.embedding_store import TalentVectorStore
from agents.feedback_agent import compute_acceptance_rate
from .llm_client import call_ollama


class AgentState(TypedDict, total=False):
    goal: str
    project_text: str
    candidates: List
    ranked_results: List

    iteration: int

    # ReAct fields
    scratchpad: List[str]
    last_observation: str
    next_action: str



store = TalentVectorStore()
MAX_ITERATIONS = 8



# def planner_node(state: AgentState):

#     iteration = state.get("iteration", 0)
#     scratchpad = state.get("scratchpad", [])

#     if iteration >= MAX_ITERATIONS:
#         return {**state, "next_action": "finish"}

#     prompt = f"""
# You are an autonomous Talent Matching Agent using ReAct reasoning.

# Goal: {state['goal']}

# Project:
# {state['project_text']}

# Previous Thoughts:
# {chr(10).join(scratchpad)}

# Last Observation:
# {state.get('last_observation', 'None')}

# Available Actions:
# - retrieve
# - score
# - rank
# - reflection
# - finish

# Follow format strictly:

# Thought: <your reasoning>
# Action: <one action from list>

# Respond EXACTLY in that format.
# """

#     response = call_ollama(prompt)

#     # Parse Thought and Action
#     thought = ""
#     action = "finish"

#     for line in response.split("\n"):
#         if line.lower().startswith("thought:"):
#             thought = line.replace("Thought:", "").strip()
#         if line.lower().startswith("action:"):
#             action = line.replace("Action:", "").strip().lower()

#     if action not in ["retrieve", "score", "rank", "reflection", "finish"]:
#         action = "finish"

#     scratchpad.append(f"Thought: {thought}")
#     scratchpad.append(f"Action: {action}")

#     return {
#         **state,
#         "scratchpad": scratchpad,
#         "next_action": action,
#         "iteration": iteration + 1
#     }





def planner_node(state: AgentState):

    iteration = state.get("iteration", 0)
    scratchpad = list(state.get("scratchpad", []))  # 🔥 copy safely

    if iteration >= MAX_ITERATIONS:
        return {**state, "next_action": "finish"}

    prompt = f"""
You are an autonomous Talent Matching Agent using ReAct reasoning.

Goal: {state['goal']}

Project:
{state['project_text']}

Previous Thoughts:
{chr(10).join(scratchpad[-10:])}

Last Observation:
{state.get('last_observation', 'None')}

Available Actions:
- retrieve
- score
- rank
- reflection
- finish

Follow format strictly:

Thought: <your reasoning>
Action: <one action from list>

Respond EXACTLY in that format.
"""

    response = call_ollama(prompt)

    thought = ""
    action = "finish"

    for line in response.split("\n"):
        line_lower = line.lower().strip()
        if line_lower.startswith("thought:"):
            thought = line.split(":", 1)[1].strip()
        if line_lower.startswith("action:"):
            action = line.split(":", 1)[1].strip().lower()

    if action not in ["retrieve", "score", "rank", "reflection", "finish"]:
        action = "finish"

    scratchpad.append(f"Thought: {thought}")
    scratchpad.append(f"Action: {action}")

    return {
        **state,
        "scratchpad": scratchpad,
        "next_action": action,
        "iteration": iteration + 1
    }





def retrieve_node(state: AgentState):

    results = store.query(state["project_text"], top_k=3)

    candidates = list(zip(
        results["ids"][0],
        results["documents"][0]
    ))

    observation = f"Retrieved {len(candidates)} candidates."

    return {
        **state,
        "candidates": candidates,
        "ranked_results": [],
        "last_observation": observation
    }




def scoring_node(state: AgentState):

    scored = []

    for emp_id, profile in state["candidates"]:
        llm_result = score_match(state["project_text"], profile)
        acceptance = compute_acceptance_rate(emp_id)
        adjusted_score = llm_result["match_score"] * acceptance

        llm_result["adjusted_score"] = adjusted_score

        scored.append({
            "emp_id": emp_id,
            "profile": profile,
            "result": llm_result
        })

    observation = f"Scored {len(scored)} candidates."

    return {
        **state,
        "ranked_results": scored,
        "last_observation": observation
    }



# def ranking_node(state: AgentState):

#     ranked = sorted(
#         state["ranked_results"],
#         key=lambda x: x["result"]["adjusted_score"],
#         reverse=True
#     )

#     top_score = ranked[0]["result"]["adjusted_score"] if ranked else 0

#     observation = f"Ranking complete. Top score: {round(top_score, 2)}"

#     return {
#         **state,
#         "ranked_results": ranked,
#         "last_observation": observation
#     }



def ranking_node(state: AgentState):

    if not state.get("ranked_results"):
        return {
            **state,
            "last_observation": "No results to rank."
        }

    ranked = sorted(
        state["ranked_results"],
        key=lambda x: x["result"].get("adjusted_score", 0),
        reverse=True
    )

    top_score = ranked[0]["result"].get("adjusted_score", 0)

    observation = f"Ranking complete. Top score: {round(top_score, 2)}"

    return {
        **state,
        "ranked_results": ranked,
        "last_observation": observation
    }





# def reflection_node(state: AgentState):

#     if not state.get("ranked_results"):
#         return {
#             **state,
#             "last_observation": "No ranked results available."
#         }

#     top_score = state["ranked_results"][0]["result"]["adjusted_score"]

#     observation = f"Reflection: Top score is {round(top_score,2)}"

#     return {
#         **state,
#         "last_observation": observation
#     }



def reflection_node(state: AgentState):

    if not state.get("ranked_results"):
        return {
            **state,
            "last_observation": "No ranked results available."
        }

    top_score = state["ranked_results"][0]["result"].get("adjusted_score", 0)

    if top_score < 0.6:
        observation = f"Top score {round(top_score,2)} too low. Need better retrieval."
    else:
        observation = f"Top score {round(top_score,2)} acceptable."

    return {
        **state,
        "last_observation": observation
    }


# -----------------------------------------------------
# GRAPH BUILDING
# -----------------------------------------------------

builder = StateGraph(AgentState)

builder.add_node("planner", planner_node)
builder.add_node("retrieve", retrieve_node)
builder.add_node("score", scoring_node)
builder.add_node("rank", ranking_node)
builder.add_node("reflection", reflection_node)

builder.set_entry_point("planner")

builder.add_conditional_edges(
    "planner",
    lambda state: state["next_action"],
    {
        "retrieve": "retrieve",
        "score": "score",
        "rank": "rank",
        "reflection": "reflection",
        "finish": END,
    }
)


# 🔁 Loop back to planner after each step
builder.add_edge("retrieve", "planner")
builder.add_edge("score", "planner")
builder.add_edge("rank", "planner")
builder.add_edge("reflection", "planner")

agentic_graph = builder.compile()



# -----------------------------------------------------
# MAIN EXECUTION FUNCTION
# -----------------------------------------------------




def run_agent(project_text: str):

    state = {
        "goal": "Find best candidates",
        "project_text": project_text,
        "candidates": [],
        "ranked_results": [],
        "iteration": 0,

        # ReAct Memory
        "scratchpad": [],
        "last_observation": "",
        "next_action": ""
    }

    final_state = agentic_graph.invoke(
        state,
        config={"recursion_limit": 100}
    )

    return final_state.get("ranked_results", [])







