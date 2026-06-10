import os
from datetime import datetime
from pymongo import MongoClient

class TrainingStatusTracker:
    """
    MongoDB-based tracker so the frontend can poll live training state safely
    across multiple backend instances.
    """

    def __init__(self, user_id: str, dataset_name: str):
        self.user_id = user_id
        self.dataset_name = dataset_name
        self.run_id = f"{user_id}_{dataset_name}"
        
        mongo_uri = os.getenv("MONGO_DB_URI")
        if not mongo_uri:
            raise ValueError("MONGO_DB_URI environment variable is not set")
        self.client = MongoClient(mongo_uri)
        self.db = self.client["MetaML"]
        self.collection = self.db["training_status"]

    def _event(self, phase: str, message: str, state: str, completed: bool = False):
        return {
            "phase": phase,
            "message": message,
            "state": state,
            "completed": completed,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

    def update(self, phase: str, message: str, completed: bool = False):
        event = self._event(
            phase=phase,
            message=message,
            state="completed" if completed else "running",
            completed=completed,
        )
        self.collection.update_one(
            {"_id": self.run_id},
            {
                "$set": {
                    "dataset": self.dataset_name,
                    "user_id": self.user_id,
                    "current": event
                },
                "$push": {"history": event}
            },
            upsert=True
        )

    def complete(self, message: str = "Training finished"):
        self.update("finished", message=message, completed=True)

    def error(self, message: str):
        event = self._event(phase="error", message=message, state="error", completed=False)
        self.collection.update_one(
            {"_id": self.run_id},
            {
                "$set": {
                    "dataset": self.dataset_name,
                    "user_id": self.user_id,
                    "current": event
                },
                "$push": {"history": event}
            },
            upsert=True
        )
