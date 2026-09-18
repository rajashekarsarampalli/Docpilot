from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .graph import graph


class AgentChatView(APIView):
    """
    API endpoint to interact with the Docpilot LangGraph Agent.
    POST: {"message": "Find my Computer Vision files"}
    """

    def post(self, request):
        user_request = (
            request.data.get("message")
            or request.data.get("user_request")
            or request.data.get("prompt")
        )

        if not user_request or not str(user_request).strip():
            return Response(
                {"error": "A 'message' or 'user_request' field is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            result = graph.invoke({
                "user_request": str(user_request).strip(),
                "user": request.user if request.user.is_authenticated else None,
            })

            return Response({
                "response": result.get("final_response"),
                "action": result.get("action"),
                "steps": result.get("steps", 0),
                "history": result.get("history", []),
            })

        except Exception as error:
            return Response(
                {"error": f"Agent processing error: {str(error)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
