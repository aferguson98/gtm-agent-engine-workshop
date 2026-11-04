import importlib
import json
import os
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult


with patch.dict(os.environ, OPENAI_API_KEY="test-key", LANGSMITH_TRACING="false"):
    agent = importlib.import_module("gtm_agent.gtm_agent")


class ProspectScoringTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, LANGSMITH_TRACING="false"))
        self.enterContext(patch.dict(agent.data_service._PROFILES, {}, clear=True))
        self.llm = self.enterContext(patch.object(agent, "_scoring_llm"))
        self.result = agent.ProspectScore(
            score=75,
            justification="Good fit with some missing technologies.",
            rubric_breakdown=agent.RubricBreakdown(
                revenue_fit=80, tech_stack_match=60, segment_fit=85
            ),
        )
        self.llm.invoke.return_value = self.result
        self.prospect_id = next(iter(agent.data_service.PROSPECTS))
        self.offering_id = next(iter(agent.data_service.OFFERINGS))
        self.ids = {"prospect_id": self.prospect_id, "offering_id": self.offering_id}

    def test_builds_and_reuses_profile(self):
        record = agent.data_service.get_prospect_record(self.prospect_id)
        profile = agent._get_or_build_profile(self.prospect_id)
        self.assertEqual(profile, {"prospect_id": self.prospect_id, **record})
        self.assertEqual(
            agent.data_service.get_profile_from_db(self.prospect_id),
            {"prospect_profile": profile},
        )
        with patch.object(agent.data_service, "get_prospect_record", return_value=None):
            self.assertEqual(agent._get_or_build_profile(self.prospect_id), profile)
            self.assertEqual(
                agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id}),
                {"prospect_profile": profile, "found": True},
            )

    def test_build_tool_delegates_to_profile_helper(self):
        for profile in ({"prospect_id": self.prospect_id}, None):
            with self.subTest(profile=profile):
                with patch.object(agent, "_get_or_build_profile", return_value=profile):
                    self.assertEqual(
                        agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id}),
                        {"prospect_profile": profile, "found": profile is not None},
                    )

    def test_missing_profile_cannot_be_built(self):
        self.assertIsNone(agent._get_or_build_profile("missing"))
        self.assertEqual(
            agent.build_prospect_profile.invoke({"prospect_id": "missing"}),
            {"prospect_profile": None, "found": False},
        )

    def test_scores_with_ids_only(self):
        self.assertEqual(set(agent.score_prospect.args), set(self.ids))
        self.assertIn("Ids are sufficient", agent.score_prospect.description)
        self.assertIn("no need to call other tools first", agent.score_prospect.description)
        self.assertEqual(agent.score_prospect.invoke(self.ids), self.result.model_dump())
        profile = agent._get_or_build_profile(self.prospect_id)
        offering = agent.data_service.get_offering(self.offering_id)
        self.llm.invoke.assert_called_once_with([
            {"role": "system", "content": agent.SCORING_PROMPT},
            {"role": "user", "content": (
                "Offering:\n" + json.dumps(offering, indent=2) +
                "\n\nProspect profile:\n" + json.dumps(profile, indent=2)
            )},
        ])

    def test_agent_executes_id_only_scoring_request(self):
        responses = [
            AIMessage(content="", tool_calls=[{
                "name": "get_current_rep", "args": {}, "id": "rep"
            }]),
            AIMessage(content="", tool_calls=[{
                "name": "score_prospect", "args": self.ids, "id": "score"
            }]),
            AIMessage(content="The prospect scores 75 out of 100."),
        ]
        with patch.object(type(agent.agent_model), "_generate", side_effect=[
            ChatResult(generations=[ChatGeneration(message=message)])
            for message in responses
        ]):
            result = agent.run_agent(
                f"Score {self.prospect_id} against {self.offering_id}",
                user_id=agent.REP_IDS[0]["rep_id"],
            )
        tool_results = [message for message in result["messages"] if isinstance(message, ToolMessage)]
        self.assertEqual([message.name for message in tool_results], ["get_current_rep", "score_prospect"])
        self.assertEqual(json.loads(tool_results[-1].content), self.result.model_dump())
        self.assertEqual(result["reply"], responses[-1].content)

    def test_missing_prospect_returns_score_null_error(self):
        self.assertEqual(
            agent.score_prospect.invoke({**self.ids, "prospect_id": "missing"}),
            {"score": None, "error": "Cannot score without a valid prospect profile."},
        )
        self.llm.invoke.assert_not_called()

    def test_missing_offering_returns_score_null_error(self):
        self.assertEqual(
            agent.score_prospect.invoke({**self.ids, "offering_id": "missing"}),
            {"score": None, "error": "Cannot score without a valid offering."},
        )
        self.llm.invoke.assert_not_called()

    def test_incomplete_offering_returns_score_null_error(self):
        offering = agent.data_service.get_offering(self.offering_id)
        for field in ("required_tech_stack", "min_annual_revenue", "description"):
            with self.subTest(field=field):
                incomplete = {key: value for key, value in offering.items() if key != field}
                with patch.object(agent.data_service, "get_offering", return_value=incomplete):
                    self.assertEqual(
                        agent.score_prospect.invoke(self.ids),
                        {"score": None, "error": "Cannot score without a valid offering."},
                    )
        self.llm.invoke.assert_not_called()

    def test_scoring_refreshes_cached_tech_stack_without_mutating_profile(self):
        profile = agent._get_or_build_profile(self.prospect_id)
        profile["tech_stack"] = ["stale"]
        current_stack = agent.data_service.fetch_tech_stack(self.prospect_id)
        self.assertEqual(agent.score_prospect.invoke(self.ids), self.result.model_dump())
        messages = self.llm.invoke.call_args.args[0]
        scored_profile = json.loads(messages[1]["content"].split("\n\nProspect profile:\n")[1])
        self.assertEqual(scored_profile, {**profile, "tech_stack": current_stack})
        self.assertEqual(profile["tech_stack"], ["stale"])


if __name__ == "__main__":
    unittest.main()
