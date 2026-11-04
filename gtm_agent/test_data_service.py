import importlib
import json
import os
import unittest
from unittest.mock import patch

from gtm_agent import data_service

with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "LANGSMITH_TRACING": "false"}):
    agent = importlib.import_module("gtm_agent.gtm_agent")


class UpdateProspectInfoTests(unittest.TestCase):
    def setUp(self):
        self.prospect_id = "LEAD-71001"
        self.record = data_service.get_prospect_record(self.prospect_id)
        self.original_stack = self.record["tech_stack"]
        self.original_profiles = dict(data_service._PROFILES)
        data_service._PROFILES.clear()
        self.addCleanup(self.restore_records)
        tracing = patch.dict(os.environ, {"LANGSMITH_TRACING": "false"})
        tracing.start()
        self.addCleanup(tracing.stop)

    def restore_records(self):
        self.record["tech_stack"] = self.original_stack
        data_service._PROFILES.clear()
        data_service._PROFILES.update(self.original_profiles)

    def test_update_persists_and_refreshes_profile(self):
        for cached in (False, True):
            with self.subTest(cached=cached):
                self.record["tech_stack"] = list(self.original_stack)
                data_service._PROFILES.clear()
                if cached:
                    profile = agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
                    self.assertNotIn("Kafka", profile["prospect_profile"]["tech_stack"])
                result = data_service.update_prospect_info(self.prospect_id, "Kafka")
                self.assertEqual(result, {
                    "updated": True, "found": True,
                    "tech_stack": self.original_stack + ["Kafka"],
                })
                self.assertIn("Kafka", data_service.fetch_tech_stack(self.prospect_id))
                profile = agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
                self.assertIn("Kafka", profile["prospect_profile"]["tech_stack"])
                data_service.update_prospect_info(self.prospect_id, "Kafka")
                self.assertEqual(data_service.fetch_tech_stack(self.prospect_id).count("Kafka"), 1)

    def test_score_uses_saved_update_even_with_old_profile(self):
        profile = agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        agent.update_prospect_info.invoke({"prospect_id": self.prospect_id, "technology": "Kafka"})
        offering = data_service.get_offering("OFFER-10005")
        with patch.object(agent, "_scoring_llm") as scoring:
            agent.score_prospect.invoke({"prospect_profile": profile["prospect_profile"], "offering": offering})
        messages = scoring.invoke.call_args.args[0]
        scored_profile = json.loads(messages[1]["content"].split("Prospect profile:\n", 1)[1])
        self.assertIn("Kafka", scored_profile["tech_stack"])
        missing = set(offering["required_tech_stack"]) - set(scored_profile["tech_stack"])
        self.assertNotIn("Kafka", missing)

    def test_unknown_prospect_is_not_updated(self):
        self.assertEqual(data_service.update_prospect_info("unknown", "Kafka"), {
            "updated": False, "found": False,
        })


if __name__ == "__main__":
    unittest.main()
