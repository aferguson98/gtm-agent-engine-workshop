"""Tests for prospect email eligibility."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "LANGSMITH_TRACING": "false"}):
    from . import gtm_agent


class SendProspectEmailTests(unittest.TestCase):
    def setUp(self):
        self.runtime = SimpleNamespace(config={"metadata": {"user_id": "rep_amills"}})
        self.prospect = {
            "prospect_id": "TEST-LEAD",
            "name": "Test Prospect",
            "email": "prospect@example.com",
        }

    def send(self, prospect):
        return gtm_agent.send_prospect_email.func(
            prospect, "Demo", "Schedule a demo", self.runtime
        )

    def test_stored_disqualification_blocks_even_if_passed_record_is_eligible(self):
        record = {**self.prospect, "disqualified": True, "email": "stored@example.com"}
        with patch.dict(gtm_agent.data_service.PROSPECTS, {"TEST-LEAD": record}):
            result = self.send({**self.prospect, "disqualified": False})
        self.assertEqual(result, {
            "status": "blocked",
            "error": "Prospect is disqualified; email not sent.",
            "to": "stored@example.com",
        })

    def test_eligible_missing_id_and_missing_record_preserve_send(self):
        record = {**self.prospect, "disqualified": False}
        without_id = {key: value for key, value in self.prospect.items() if key != "prospect_id"}
        prospects = [self.prospect, without_id, {**self.prospect, "prospect_id": "MISSING"}]
        with patch.dict(gtm_agent.data_service.PROSPECTS, {"TEST-LEAD": record}):
            for prospect in prospects:
                with self.subTest(prospect=prospect):
                    result = self.send(prospect)
                    message_id = result.pop("message_id")
                    self.assertTrue(message_id.startswith("msg-"))
                    self.assertEqual(result, {
                        "status": "sent",
                        "to": "prospect@example.com",
                        "to_name": "Test Prospect",
                        "from": "ava.mills@northpoint.com",
                        "from_name": "Ava Mills",
                        "subject": "Demo",
                        "body": "Schedule a demo",
                    })

    def test_missing_email_preserves_failure(self):
        self.assertEqual(self.send({}), {
            "status": "failed",
            "error": "Prospect record has no email address.",
        })

    def test_explicit_sender_is_preserved(self):
        result = gtm_agent.send_prospect_email.func(
            self.prospect, "Demo", "Schedule a demo", self.runtime,
            from_rep={"name": "Other Rep", "email": "rep@example.com"},
        )
        self.assertEqual(result["from"], "rep@example.com")
        self.assertEqual(result["from_name"], "Other Rep")


if __name__ == "__main__":
    unittest.main()
