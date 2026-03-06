from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional


class ActionType(str, Enum):
    NO_OP = "no_op"
    SCALE_OUT = "scale_out"
    SCALE_IN = "scale_in"
    MIGRATE = "migrate"
    PLACE = "place"


class AgentAction(BaseModel):
    action_type: ActionType

    # Targets
    app_id: Optional[str] = None
    service_id: Optional[str] = None

    # For Scaling #(delta_replicas)
    magnitude: int = Field(default=0, ge=0)

    # For Migration/Placement (destination)
    target_cluster: Optional[str] = None

    def __str__(self):
        if self.action_type == ActionType.NO_OP:
            return "NO-OP"
        return f"{self.action_type.value.upper()} {self.service_id} (Mag: {self.magnitude}, Dest: {self.target_cluster})"