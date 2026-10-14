from __future__ import annotations

import copy
import json
import unittest

from .verifier import ANSWER_POLICY_VERSION, verify_answer


class VerifierTests(unittest.TestCase):
    def check_status(self, answer_type: str, gold: str, prediction: str, status: str, *, finished=True, **record_fields):
        record = {"answer_type": answer_type, "answer": gold, **record_fields}
        result = verify_answer(record, prediction, finished=finished)
        self.assertEqual(result["status"], status, result)
        json.dumps(result, ensure_ascii=False)
        return result

    def test_policy_version_and_pure_json_result(self):
        self.assertEqual(ANSWER_POLICY_VERSION, "provisional-v3-verifier-semantics")
        record = {"answer_type": "integer", "answer": "3", "extra": [1, 2]}
        original = copy.deepcopy(record)
        result = verify_answer(record, "3")
        self.assertEqual(record, original)
        json.dumps(result)

    def test_integer_strict_boundaries(self):
        self.check_status("integer", "-3", "−3", "invalid")
        self.check_status("integer", "+003", "3", "correct")
        self.check_status("integer", "3", "3 trailing", "invalid")
        self.check_status("integer", "3", "3.0", "invalid")
        self.check_status("integer", "3", "٣", "invalid")
        self.check_status("integer", "03", "3", "correct")
        self.check_status("integer", "not-an-int", "3", "invalid")

    def test_decimal_exact_syntax_and_finiteness(self):
        self.check_status("decimal", "1.250", "1.25", "correct")
        self.check_status("decimal", "-0.5", "-0.50", "correct")
        for malformed in ("NaN", "Infinity", "1e3", "1_000", "01.2", ".5", "1.", "1.2x"):
            with self.subTest(malformed=malformed):
                self.check_status("decimal", "1.2", malformed, "invalid")
        self.check_status("decimal", "NaN", "1", "invalid")

    def test_fraction_exactness_and_denominator(self):
        self.check_status("fraction", "1/2", "2/4", "correct")
        self.check_status("fraction", "1/2", "3/4", "incorrect")
        self.check_status("fraction", "1/2", "1/0", "invalid")
        self.check_status("fraction", "1/2", "1 / 2", "invalid")
        self.check_status("fraction", "1/0", "1/2", "invalid")

    def test_boolean_uses_only_case_insensitive_table(self):
        self.check_status("boolean", "true", "YES", "correct")
        self.check_status("boolean", "no", "false", "correct")
        self.check_status("boolean", "yes", "1", "invalid")
        self.check_status("boolean", "perhaps", "yes", "invalid")

    def test_number_list_mixed_exact_values_keep_order(self):
        self.check_status("number_list", "1, 1/2, 0.25, -2", "1, 2/4, 0.250, -2", "correct")
        self.check_status("number_list", "1, 2", "2, 1", "incorrect")
        self.check_status("number_list", "1, 2", "1, 3", "incorrect")

    def test_number_list_rejects_malformed_or_injected_text(self):
        for malformed in ("", ",1", "1,", "1,,2", "(1, 2)", "1, [2]", "1, two", "1, 2 answer"):
            with self.subTest(malformed=malformed):
                self.check_status("number_list", "1, 2", malformed, "invalid")
        self.check_status("number_list", "1,,2", "1, 2", "invalid")

    def test_base_number_reads_only_explicit_question_radix(self):
        self.check_status(
            "base_number", "241a", "241A", "correct",
            question="In base 13, what is 241c + -2?",
        )
        self.check_status(
            "base_number", "5c", "5c", "correct",
            question="What is 232 (base 6) in base 16?",
        )
        self.check_status(
            "base_number", "a99a", "a99a", "correct",
            question="In base 12, what is a99b + -1?",
        )
        self.check_status(
            "base_number", "5c", "5g", "invalid",
            question="What is 232 (base 6) in base 16?",
        )
        self.check_status(
            "base_number", "5c", "05c", "invalid",
            question="What is 232 (base 6) in base 16?",
        )
        self.check_status("base_number", "10", "10", "unsupported", question="Compute 8 + 2")
        self.check_status(
            "base_number", "2", "2", "invalid",
            question="In base 2, what is 10 + 1?",
        )

    def test_time_grammar_and_exact_time_of_day(self):
        self.check_status("time", "8:03 PM", "08:03 pm", "invalid")
        self.check_status("time", "8:03 PM", "8:03 pm", "correct")
        self.check_status("time", "12:00 AM", "12:00 PM", "incorrect")
        self.check_status("time", "9:09 PM", "13:09 PM", "invalid")
        self.check_status("time", "9:09 PM", "9:60 PM", "invalid")
        self.check_status("time", "9:09 PM", "9:09PM", "invalid")
        self.check_status("time", "24:00", "9:09 PM", "invalid")

    def test_exact_strings_unicode_normalization_and_unsupported(self):
        self.check_status("symbol", "café", "cafe\u0301", "correct")
        self.check_status("option_choice", "A", " A\u00a0", "correct")
        self.check_status("expression", "x", "__import__('os').system('x')", "unsupported")
        self.check_status("symbolic_list", "x, 1", "x, 1", "unsupported")

    def test_empty_protocol_and_unfinished_generation(self):
        self.check_status("integer", "1", "", "invalid")
        self.check_status("integer", "1", "<ANSWER> 1 <EOS>", "invalid")
        self.check_status("integer", "1", "1", "invalid", finished=False)
        result = verify_answer({"answer_type": "integer", "answer": "1"}, "1", finished=False)
        self.assertIn("EOS", result["reason"])


if __name__ == "__main__":
    unittest.main()
