from dataclasses import dataclass
from typing import Literal

IndexingStatus = Literal["idle", "running", "done", "error"]


@dataclass
class IndexingState:
    status: IndexingStatus = "idle"
    message: str = ""


indexing_state = IndexingState()


def set_status(status: IndexingStatus, message: str = "") -> None:
    indexing_state.status = status
    indexing_state.message = message
