from openai import OpenAI
import os
import time

from .tools import (
    AllStep,
    ActionStep,
    TerminateStep
)

class Reasoner:
    def __init__(self, model_type, model_name_or_path, port):
        self.model_type = model_type
        self.model_name_or_path = model_name_or_path
        self.model_port = port
        if self.model_type == "gpt-4o":
            openai_base_url = os.environ.get("OPENAI_BASE_URL")
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            self.model = OpenAI(
                api_key=openai_api_key,
                base_url=openai_base_url
            )
        elif self.model_type == "qwen-vl":
            import os
            os.environ["http_proxy"] = ""
            os.environ["https_proxy"] = ""
            openai_api_key = "EMPTY"
            openai_api_base = f"http://localhost:{self.model_port}/v1"
            self.model = OpenAI(api_key=openai_api_key,base_url=openai_api_base)
            
        
    def get_completion(self, message_list, step_type="all", retry=5, timeout=3):
        response = None
        if self.model_type == 'gpt-4o':
            if step_type == "all":
                response_format = {
                    "type": "json_schema", 
                    "json_schema": {
                        "name": "AllStep",
                        "schema": AllStep.model_json_schema()
                        }
                }
            elif step_type == "action":
                response_format = {
                    "type": "json_schema", 
                    "json_schema": {
                        "name": "ActionStep",
                        "schema": ActionStep.model_json_schema()
                        }
                }
            elif step_type == "terminate":
                response_format = {
                    "type": "json_schema", 
                    "json_schema": {
                        "name": "TerminateAction",
                        "schema": TerminateStep.model_json_schema()
                        }
                }
            chat_params = {
                "model": "gpt-4o",
                "messages": message_list,
                "response_format": response_format,
                "temperature": 0.0
            }
            for i in range(retry):
                try:
                    response = self.model.chat.completions.create(**chat_params)
                    break
                except Exception as e:
                    print(f"Error when geting complement in {i}th tries: {e}")
                    response = None
                    time.sleep(timeout)    
        elif self.model_type == "qwen-vl":
            chat_params = {
                "model": self.model_name_or_path,
                "messages": message_list,
                "temperature": 0.0
            }
            for i in range(retry):
                try:
                    response = self.model.chat.completions.create(**chat_params)
                    break
                except Exception as e:
                    print(f"Error when geting complement in {i}th tries: {e}")
                    response = None
                    time.sleep(timeout)
        return response
       