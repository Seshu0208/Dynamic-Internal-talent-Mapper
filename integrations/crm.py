import os
import requests


# class CRMAdapter:
#     def fetch_projects(self):
#         return requests.get("CRM_API").json()
    

class CRMIntegration:

    def push_candidate_recommendation(self, project_id, emp_id, score):
        print(f"CRM Updated → Project {project_id}, Employee {emp_id}, Score {score}")