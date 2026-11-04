import importlib
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage


with patch.dict("os.environ", {"OPENAI_API_KEY": "test", "LANGSMITH_TRACING": "false"}):
    module = importlib.import_module("gtm_agent.gtm_agent")


class RunAgentTests(unittest.TestCase):
    def test_rep_context_precedes_email_and_scoring_requests(self):
        rep = module.data_service.REP_IDS[0]
        for request in ("Email LEAD-12853", "Score LEAD-12853 for OFFER-10001"):
            with self.subTest(request=request):
                agent = Mock()
                agent.invoke.return_value = {"messages": [AIMessage(content="Done")]}
                with patch.object(module, "gtm_agent", agent):
                    result = module.run_agent(request, user_id=rep["rep_id"])
                messages = agent.invoke.call_args.args[0]["messages"]
                self.assertEqual([message["role"] for message in messages], ["system", "user"])
                self.assertIn(rep["name"], messages[0]["content"])
                self.assertIn(rep["email"], messages[0]["content"])
                self.assertIn("Sign emails", messages[0]["content"])
                self.assertEqual(messages[1]["content"], request)
                self.assertEqual(result["reply"], "Done")
                self.assertEqual(agent.invoke.call_args.kwargs["config"]["metadata"]["user_id"], rep["rep_id"])

    def test_unknown_rep_preserves_user_request(self):
        agent = Mock()
        agent.invoke.return_value = {"messages": [AIMessage(content="Done")]}
        with patch.object(module, "gtm_agent", agent):
            module.run_agent("Score LEAD-12853", user_id="unknown")
        self.assertEqual(agent.invoke.call_args.args[0]["messages"], [
            {"role": "user", "content": "Score LEAD-12853"},
        ])

    def test_email_sender_still_defaults_to_runtime_identity(self):
        rep = module.data_service.REP_IDS[1]
        runtime = SimpleNamespace(config={"metadata": {"user_id": rep["rep_id"]}})
        body = f"Hello,\n\nBest,\n{rep['name']}"
        result = module.send_prospect_email.func(
            {"name": "Prospect", "email": "prospect@example.com"},
            "Introduction", body, runtime,
        )
        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["from"], rep["email"])
        self.assertEqual(result["from_name"], rep["name"])
        self.assertEqual(result["body"], body)

    def test_identity_tool_remains_available_without_mandatory_lookup(self):
        rep = module.data_service.REP_IDS[0]
        runtime = SimpleNamespace(config={"metadata": {"user_id": rep["rep_id"]}})
        tools = module.gtm_agent.get_graph().nodes["tools"].data.tools_by_name
        self.assertIn("get_current_rep", tools)
        self.assertEqual(tools["get_current_rep"].func(runtime), {"rep": rep, "found": True})
        self.assertNotIn("get_current_rep", module.SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
