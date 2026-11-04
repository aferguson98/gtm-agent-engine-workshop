"""Tests for prospect data exposed by the agent tools."""

import importlib
import json
import unittest
from unittest.mock import patch

with patch("dotenv.load_dotenv"), patch("langchain_openai.ChatOpenAI"), \
        patch("deepagents.create_deep_agent"):
    agent = importlib.import_module("gtm_agent.gtm_agent")


class ProspectToolsTest(unittest.TestCase):
    def setUp(self):
        self.prospect_id = "LEAD-TEST"
        self.record = {
            "name": "Test Prospect",
            "email": "prospect@example.com",
            "annual_revenue": 50000000,
            "enrichment_source": "Test enrichment",
            "disqualified": False,
            "engagement_history": [{"channel": "Webinar"}],
            "account_details": [{"account": "Test account"}],
            "tech_stack": ["AWS"],
            "billing_qualification": {"tax_id": "private-billing-value"},
            "unapproved_field": "private-extra-value",
        }
        self.enterContext(patch.dict("os.environ", {"LANGSMITH_TRACING": "false"}))
        self.enterContext(patch.object(agent.data_service, "PROSPECTS", {
            self.prospect_id: self.record,
        }))
        self.enterContext(patch.object(agent.data_service, "_PROFILES", {}))

    def test_contact_contains_only_approved_fields(self):
        self.assertEqual(agent.get_prospect.invoke({"prospect_id": self.prospect_id}), {
            "prospect": {
                "prospect_id": self.prospect_id,
                "name": self.record["name"],
                "email": self.record["email"],
            },
            "found": True,
        })

    def test_profile_and_cache_contain_only_approved_fields(self):
        expected = {k: v for k, v in self.record.items()
                    if k not in ("billing_qualification", "unapproved_field")}
        expected["prospect_id"] = self.prospect_id
        for _ in range(2):
            self.assertEqual(agent.build_prospect_profile.invoke({
                "prospect_id": self.prospect_id,
            }), {"prospect_profile": expected, "found": True})
        self.assertEqual(agent.data_service.get_profile_from_db(self.prospect_id), {
            "prospect_profile": expected,
        })

    def test_missing_optional_fields_remain_absent(self):
        for key in ("name", "email", "annual_revenue", "enrichment_source", "disqualified"):
            del self.record[key]
        self.assertEqual(agent.get_prospect.invoke({"prospect_id": self.prospect_id}), {
            "prospect": {"prospect_id": self.prospect_id}, "found": True,
        })
        expected = {k: self.record[k] for k in (
            "engagement_history", "account_details", "tech_stack",
        )}
        expected["prospect_id"] = self.prospect_id
        self.assertEqual(agent.build_prospect_profile.invoke({
            "prospect_id": self.prospect_id,
        }), {"prospect_profile": expected, "found": True})

    def test_unknown_prospect_preserves_not_found_results(self):
        self.assertEqual(agent.get_prospect.invoke({"prospect_id": "missing"}), {
            "prospect": None, "found": False,
        })
        self.assertEqual(agent.build_prospect_profile.invoke({"prospect_id": "missing"}), {
            "prospect_profile": None, "found": False,
        })

    def test_scoring_removes_billing_and_preserves_result(self):
        offering = {
            "required_tech_stack": ["AWS"],
            "min_annual_revenue": 1000000,
            "description": "Cloud platform",
        }
        expected = agent.ProspectScore(
            score=85, justification="Good fit",
            rubric_breakdown=agent.RubricBreakdown(
                revenue_fit=90, tech_stack_match=100, segment_fit=65,
            ),
        )
        built = agent.build_prospect_profile.invoke({
            "prospect_id": self.prospect_id,
        })["prospect_profile"]
        for profile in (built, {**built, "billing_qualification": self.record["billing_qualification"]},
                        {"billing_qualification": self.record["billing_qualification"]}):
            with self.subTest(profile_keys=list(profile)), patch.object(agent, "_scoring_llm") as model:
                model.invoke.return_value = expected
                result = agent.score_prospect.invoke({"prospect_profile": profile, "offering": offering})
                self.assertEqual(result, expected.model_dump())
                prompt = model.invoke.call_args.args[0][1]["content"]
                sent_profile = json.loads(prompt.split("Prospect profile:\n", 1)[1])
                self.assertEqual(sent_profile, {k: v for k, v in profile.items()
                                                if k != "billing_qualification"})
                self.assertNotIn("private-billing-value", prompt)
        self.assertIn("billing_qualification", self.record)


if __name__ == "__main__":
    unittest.main()
