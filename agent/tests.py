from django.test import TestCase
from django.contrib.auth.models import User
from rest_framework.test import APIClient
from rest_framework import status

from agent.tools import normalize_parameters, execute_tool, TOOLS
from agent.llm import parse_planner_decision, clean_response


class AgentUnitTestCase(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="agentuser", password="password123")
        self.client = APIClient()

    def test_clean_response(self):
        text = "<think>internal thinking</think>{\"action\":\"list_files\",\"parameters\":{}}"
        cleaned = clean_response(text)
        self.assertEqual(cleaned, "{\"action\":\"list_files\",\"parameters\":{}}")

        markdown_json = "```json\n{\"action\":\"list_files\",\"parameters\":{}}\n```"
        self.assertEqual(clean_response(markdown_json), "{\"action\":\"list_files\",\"parameters\":{}}")

    def test_parse_planner_decision(self):
        valid = '{"action": "search_files", "parameters": {"query": "CV"}}'
        decision = parse_planner_decision(valid)
        self.assertEqual(decision["action"], "search_files")
        self.assertEqual(decision["parameters"]["query"], "CV")

        with self.assertRaises(ValueError):
            parse_planner_decision('{"action": "fake_unknown_action", "parameters": {}}')

    def test_normalize_parameters(self):
        norm = normalize_parameters("create_folder", {"name": "TestDir"})
        self.assertEqual(norm["folder_name"], "TestDir")

        norm_zip = normalize_parameters("create_zip", {"file_id": 10, "archive": "myzip"})
        self.assertEqual(norm_zip["file_ids"], [10])
        self.assertEqual(norm_zip["zip_name"], "myzip")

    def test_agent_chat_api_auth(self):
        # Unauthenticated request should be rejected
        res_unauth = self.client.post("/api/agent/chat/", {"message": "hello"})
        self.assertIn(res_unauth.status_code, [status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN])

        # Authenticated empty message
        self.client.force_authenticate(user=self.user)
        res_empty = self.client.post("/api/agent/chat/", {"message": ""})
        self.assertEqual(res_empty.status_code, status.HTTP_400_BAD_REQUEST)

    def test_execute_tool(self):
        self.assertIn("list_files", TOOLS)
        res = execute_tool("list_files", {}, user=self.user)
        self.assertIsInstance(res, list)
