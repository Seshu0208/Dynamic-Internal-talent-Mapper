import os
import requests


# class HRMSAdapter:
#     def get_employee(self, emp_id):
#         return requests.get(f"HRMS_API/{emp_id}").json()
    

class HRMSIntegration:

    def get_employee_availability(self, emp_id):
        # Stub
        return "Available"

    def get_employee_location(self, emp_id):
        return "Bangalore"