from typing import TypedDict, Any, Optional


class AgentState(TypedDict, total=False):
    user_request: str
    action: str
    parameters: dict
    tool_result: Any
    final_response: str
    steps: int
    history: list
    user: Optional[Any]  # authenticated Django user instance
    download_path: Optional[str]  # file path for Telegram download