from pydantic import BaseModel
from enum import Enum


class GuardrailFlag(Enum):
    NORMAL = "NORMAL"
    MALICIOUS = "MALICIOUS"


class GuardrailsOutput(BaseModel):
    flag: GuardrailFlag
    justification: str
