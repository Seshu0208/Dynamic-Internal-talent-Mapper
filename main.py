# from agents.agentic_orchestrator import run_agent
# from tools.file_ingestion import extract_text_from_file
# from agents.agentic_orchestrator import store
# from agents.skill_extraction_agent import extract_structured_skills
# import os

from agents.agentic_orchestrator_up1 import run_agent
from tools.file_ingestion import extract_text_from_file
from agents.agentic_orchestrator_up1 import store
from agents.skill_extraction_agent import extract_structured_skills
import os


path = r"/home/gumduboinaseshubabu/Downloads/cv_profiles_db"
#path = r"/home/gumduboinaseshubabu/Downloads/single_cv"

project_text = """ 
    Here are the core responsibilities and technologies they should be aware of:
        Hands-on experience with CI/CD pipelines using Jenkins
        Expert knowledge of Git version control with GitHub Pipeline and Bitbucket
        Practical experience in containerization using Docker
        Design and maintain CI/CD pipelines with Python.
        Automate infrastructure and manage containerized applications.
        Collaborate with development teams to streamline deployment processes.
    """



if __name__ == "__main__":

    for profile in os.listdir(path):
        profilepath = os.path.join(path,profile)
        print(profilepath)
        emp_id = profilepath.split('/')[-1].split('.')[0].split("_")[-1]
        print("Employee ID : ",emp_id, "\n")
        #txt = extract_text_from_file(profilepath)

        # 1️⃣ Index employees first
        emp_text = extract_text_from_file(profilepath)

        # ✅ EXTRACT STRUCTURED SKILLS
        structured = extract_structured_skills(emp_text)

        store.add_employee(emp_id, emp_text, metadata=structured)

        print("Collection count:", store.collection.count())

        # 2️⃣ Run agent
        # project_text = extract_text_from_file("project.docx")

    results = run_agent(project_text)

    print("FINAL RESULTS:", results)

    


