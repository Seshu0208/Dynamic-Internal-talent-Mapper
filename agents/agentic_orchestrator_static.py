# agents/agentic_orchestrator.py

from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from agents.matcher_agent import score_match
from agents.skill_extraction_agent import extract_structured_skills
from tools.embedding_store import TalentVectorStore
from agents.feedback_agent import compute_acceptance_rate


# -----------------------------------------------------
# STATE DEFINITION
# -----------------------------------------------------

class AgentState(TypedDict):
    goal: str
    project_text: str
    candidates: List
    ranked_results: List
    step: str


store = TalentVectorStore()


# -----------------------------------------------------
# 1. PLANNER NODE (LLM THINKING)
# -----------------------------------------------------

def planner_node(state: AgentState):

    if state.get("step") is None:
        return {**state, "step": "retrieve_candidates"}

    if state["step"] == "retrieve_candidates":
        return {**state, "step": "score_candidates"}

    if state["step"] == "score_candidates":
        return {**state, "step": "rank_candidates"}

    return state


# -----------------------------------------------------
# 2. RETRIEVE NODE
# -----------------------------------------------------

def retrieve_node(state: AgentState):

    results = store.query(state["project_text"], top_k=5)

    candidates = list(zip(
        results["ids"][0],
        results["documents"][0]
    ))

    return {**state, "candidates": candidates}


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

    return {**state, "ranked_results": scored}


# -----------------------------------------------------
# 4. RANKING NODE
# -----------------------------------------------------

def ranking_node(state: AgentState):

    ranked = sorted(
        state["ranked_results"],
        key=lambda x: x["result"]["adjusted_score"],
        reverse=True
    )

    return {**state, "ranked_results": ranked}


# -----------------------------------------------------
# GRAPH BUILDING
# -----------------------------------------------------

builder = StateGraph(AgentState)

builder.add_node("planner", planner_node)
builder.add_node("retrieve", retrieve_node)
builder.add_node("score", scoring_node)
builder.add_node("rank", ranking_node)

builder.set_entry_point("planner")

builder.add_edge("planner", "retrieve")
builder.add_edge("retrieve", "score")
builder.add_edge("score", "rank")
builder.add_edge("rank", END)

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
        "step": None
    }

    final_state = agentic_graph.invoke(state)

    return final_state["ranked_results"]


