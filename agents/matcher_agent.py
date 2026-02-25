import json
from .llm_client import call_ollama
from .skill_graph import SKILL_GRAPH


# -------------------------------------------------------
# 1. APPLY TRANSFERABLE SKILL BOOST
# -------------------------------------------------------

def apply_transferable_boost(project_text: str,
                             employee_text: str,
                             base_score: float) -> float:
    """
    If project requires a skill in SKILL_GRAPH,
    and employee has related skills,
    boost the score proportionally.
    """

    boosted_score = base_score

    for skill, config in SKILL_GRAPH.items():

        # Check if main skill is required in project
        if skill.lower() in project_text.lower():

            related_skills = config["related_skills"]
            boost_weight = config["boost_weight"]

            # Count related skill matches
            matched = sum(
                1 for s in related_skills
                if s.lower() in employee_text.lower()
            )

            if len(related_skills) > 0:
                boost = (matched / len(related_skills)) * boost_weight
                boosted_score += boost

    # Cap at 1.0
    return min(boosted_score, 1.0)


# -------------------------------------------------------
# 2. MAIN MATCH FUNCTION
# -------------------------------------------------------

def score_match(project_text: str, employee_text: str) -> dict:
    """
    Uses LLM to compute base match score,
    then applies transferable skill boost.
    """

    prompt = f"""
You are an AI Talent Matching Agent.

PROJECT REQUIREMENTS:
{project_text}

EMPLOYEE PROFILE:
{employee_text}

Instructions:
- Calculate match score between 0.0 and 1.0
- Identify strengths
- Identify missing skills
- Consider transferable skills
- Provide recommendation

Return STRICT JSON:

{{
  "match_score": 0.0,
  "strengths": [],
  "missing_skills": [],
  "transferable_skills_reasoning": "",
  "final_recommendation": ""
}}
"""

    response = call_ollama(prompt)

    # -------------------------------------------------------
    # Extract JSON safely
    # -------------------------------------------------------
    try:
        start = response.find("{")
        end = response.rfind("}") + 1
        parsed_json = json.loads(response[start:end])
    except Exception:
        return {
            "match_score": 0.0,
            "strengths": [],
            "missing_skills": [],
            "transferable_skills_reasoning": "Parsing failed",
            "final_recommendation": "Not Recommended"
        }

    # -------------------------------------------------------
    # Apply transferable skill boost
    # -------------------------------------------------------

    base_score = parsed_json.get("match_score", 0.0)

    # boosted_score = apply_transferable_boost(
    #     project_text,
    #     employee_text,
    #     base_score
    # )

    parsed_json["match_score"] = base_score   #boosted_score

    return parsed_json



