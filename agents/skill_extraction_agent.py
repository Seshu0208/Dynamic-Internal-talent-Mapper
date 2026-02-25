from .llm_client import call_ollama
import json
import time

def extract_structured_skills(text):
    prompt = f"""
Extract structured skill data from this resume or job description.

Return JSON:
{{
  "primary_skills": [],
  "secondary_skills": [],
  "tools": [],
  "experience_years": {{}}
}}

TEXT:
{text}
""" 
    starttime = time.time()
    response = call_ollama(prompt)
    endtime = time.time()
    print("execution time : ",endtime-starttime)

    try:
        start = response.find("{")
        end = response.rfind("}") + 1
        return json.loads(response[start:end])
    except Exception:
        return {
            "primary_skills": [],
            "secondary_skills": [],
            "tools": [],
            "experience_years": {}
        }


