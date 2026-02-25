# agents/agentic_orchestrator.py

from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from agents.matcher_agent import score_match
from agents.skill_extraction_agent import extract_structured_skills
from tools.embedding_store import TalentVectorStore
from agents.feedback_agent import compute_acceptance_rate
from .llm_client import call_ollama

# -----------------------------------------------------
# STATE DEFINITION
# -----------------------------------------------------

class AgentState(TypedDict, total=False):
    goal: str
    project_text: str
    candidates: List
    ranked_results: List

    # Control fields
    # next_action: str
    step: str
    iteration: int
    retried: bool


store = TalentVectorStore()


# -----------------------------------------------------
# 1. PLANNER NODE (LLM THINKING)
# -----------------------------------------------------


MAX_ITERATIONS = 8


def planner_node(state):

    iteration = state.get("iteration", 0)

    if iteration >= MAX_ITERATIONS:
        return {**state, "next_action": "finish"}

    if not state.get("candidates"):
        return {**state, "next_action": "retrieve"}

    # Score if not yet scored
    if state.get("candidates") and not state.get("ranked_results"):
        return {**state, "next_action": "score"}

    # 🔥 ADD THIS BLOCK
    if state.get("ranked_results") and not state.get("rank_complete"):
        return {**state, "next_action": "rank"}

    if state.get("rank_complete") and not state.get("reflection_done"):
        return {**state, "next_action": "reflection"}

    return {**state, "next_action": "finish"}



# -----------------------------------------------------
# 2. RETRIEVE NODE
# -----------------------------------------------------

def retrieve_node(state: AgentState):

    results = store.query(state["project_text"], top_k=5)

    candidates = list(zip(
        results["ids"][0],
        results["documents"][0]
    ))

    return {
        **state,
        "candidates": candidates,
        "ranked_results": [],   # 🔥 CLEAR OLD RESULTS
        "iteration": state.get("iteration", 0) + 1
    }


# -----------------------------------------------------
# 3. SCORING NODE
# -----------------------------------------------------

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

    return {**state, "ranked_results": scored, "iteration": state.get("iteration", 0) + 1}


# -----------------------------------------------------
# 4. RANKING NODE
# -----------------------------------------------------


def ranking_node(state: AgentState):

    ranked = sorted(
        state["ranked_results"],
        key=lambda x: x["result"]["adjusted_score"],
        reverse=True
    )

    return {
        **state,
        "ranked_results": ranked,
        "rank_complete": True,
        "iteration": state.get("iteration", 0) + 1
    }

    ##  "ranked_sorted": True,  # 🔥 add this

# -----------------------------------------------------
# 4. REFLECTION NODE
# -----------------------------------------------------


def reflection_node(state):

    iteration = state.get("iteration", 0)

    if iteration >= MAX_ITERATIONS:
        return {**state, "next_action": "finish"}

    if not state.get("ranked_results"):
        return {**state, "next_action": "finish"}

    top_score = state["ranked_results"][0]["result"]["adjusted_score"]

    # Only retry once
    if not state.get("reflection_done") and top_score < 0.7:
        return {
            **state,
            "next_action": "retrieve",
            "reflection_done": True,
            "rank_complete": False,
            "ranked_results": [],
            "iteration": iteration + 1
        }

    return {
        **state,
        "reflection_done": True,
        "next_action": "finish",
        "iteration": iteration + 1
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
    "rank_complete": False,
    "reflection_done": False
}

    final_state = agentic_graph.invoke(state, config={"recursion_limit": 500})

    return final_state.get("ranked_results", [])


