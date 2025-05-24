import os
from dotenv import load_dotenv
from openai import OpenAI

"""
This module defines the `NeoAssistant` class to interact with the OpenAI API Beta, 
used for its thread-level context support. The assistant, N.E.O., performs tasks 
like scheduling, web searches, and general assistance with British sass.
"""


class AIAssistant:
    def __init__(self, name, model, description):
        load_dotenv()

        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key)
        
        self._assistant_name = name
        self._model = model
        self._model_description = description
        self._thread_id = None

    def _create_assistant(self):

        confirm = input("Do you want to create a new assistant? (yes/no): ").strip().lower()

        if confirm != "yes":
            print("Aborting assistant creation.")
            return None
        
        neo_model = self.client.beta.assistants.create(
            name=self._assistant_name,
            model=self._model,
            instructions=self._model_description,
            tools=[{"type": "code_interpreter"}],
        )

        if neo_model:
            print(f"Assistant created with ID: {neo_model.id}")
        else:
            raise Exception("No assistants found and creation failed.")

        return neo_model

    def _list_assistants(self):
        assistants = self.client.beta.assistants.list(
            order="desc",
            limit="20",
        )

        if assistants and assistants.data:
            print("Assistants were found.")
        else:
            print("No assistants found.")
        
        return assistants

    def _get_assistant_id(self):

        assistants = self._list_assistants()
        neo_model = None

        if assistants and assistants.data:
            for assistant in assistants.data:
                if (assistant.name == self._assistant_name and assistant.model == self._model):
                    print("Assistant already exists. Skipping creation.")
                    return assistant.id

        neo_model = self._create_assistant()
        return neo_model.id
    
    def _ensure_thread(self):
        if not self._thread_id:
            thread = self.client.beta.threads.create()
            self._thread_id = thread.id

        print("We will be using this theead ID:", self._thread_id)
    
    def talk_to_assistant(self, message):
        self._ensure_thread()

        self.client.beta.threads.messages.create(
            thread_id=self._thread_id,
            role="user",
            content=message,
        )

        run = self.client.beta.threads.runs.create(
            assistant_id=self._get_assistant_id(),
            thread_id=self._thread_id
        )

        while True:
            run_status = self.client.beta.threads.runs.retrieve(
                thread_id=self._thread_id, run_id=run.id
            )
            if run_status.status == "completed":
                break
            elif run_status.status in ["failed", "cancelled", "expired"]:
                raise Exception(f"Run failed: {run_status.status}")
        
        messages = self.client.beta.threads.messages.list(thread_id=self._thread_id)

        for msg in messages.data:
            if msg.role == "assistant":
                return msg.content[0].text.value

        return None


neo = AIAssistant("Neo", "gpt-4o-mini", "Your assistant for scheduling, web searches, and general assistance with British sass.")

print(neo.talk_to_assistant("what do you mean we are having a deja vu? How many times have we done this?"))
