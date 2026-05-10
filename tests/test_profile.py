import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.memory_agent import ensure_profile_structure


REQUIRED_KEYS = [
    "sessions", "questions_asked", "last_level", "topics_seen",
    "level_history", "topic_counts", "weak_areas", "mastery",
    "used_explanations", "recommended_next_topics", "last_evaluation",
]


class TestEnsureProfileStructure(unittest.TestCase):
    def test_empty_dict_gets_all_defaults(self):
        p = ensure_profile_structure({})
        for key in REQUIRED_KEYS:
            self.assertIn(key, p, f"Missing key: {key}")

    def test_none_input_treated_as_empty(self):
        p = ensure_profile_structure(None)
        self.assertIsInstance(p, dict)
        for key in REQUIRED_KEYS:
            self.assertIn(key, p)

    def test_default_values_are_correct_types(self):
        p = ensure_profile_structure({})
        self.assertIsInstance(p["sessions"], int)
        self.assertIsInstance(p["questions_asked"], int)
        self.assertIsInstance(p["last_level"], str)
        self.assertIsInstance(p["topics_seen"], list)
        self.assertIsInstance(p["level_history"], list)
        self.assertIsInstance(p["topic_counts"], dict)
        self.assertIsInstance(p["weak_areas"], dict)
        self.assertIsInstance(p["mastery"], dict)
        self.assertIsInstance(p["used_explanations"], dict)
        self.assertIsInstance(p["recommended_next_topics"], list)

    def test_existing_values_preserved(self):
        p = ensure_profile_structure({
            "sessions": 7,
            "last_level": "advanced",
            "mastery": {"transformers": 0.85},
        })
        self.assertEqual(p["sessions"], 7)
        self.assertEqual(p["last_level"], "advanced")
        self.assertEqual(p["mastery"]["transformers"], 0.85)

    def test_missing_keys_filled_without_losing_existing(self):
        p = ensure_profile_structure({"sessions": 3})
        self.assertEqual(p["sessions"], 3)
        self.assertEqual(p["questions_asked"], 0)

    def test_idempotent(self):
        p1 = ensure_profile_structure({})
        p2 = ensure_profile_structure(p1)
        self.assertEqual(p1, p2)

    def test_old_weak_areas_list_converted_to_dict(self):
        p = ensure_profile_structure({"weak_areas": ["topic1", "topic2"]})
        self.assertIsInstance(p["weak_areas"], dict)
        self.assertIn("topic1", p["weak_areas"])
        self.assertIn("topic2", p["weak_areas"])

    def test_corrupt_type_for_mastery_replaced(self):
        p = ensure_profile_structure({"mastery": "not_a_dict"})
        self.assertIsInstance(p["mastery"], dict)

    def test_corrupt_type_for_topics_seen_replaced(self):
        p = ensure_profile_structure({"topics_seen": 42})
        self.assertIsInstance(p["topics_seen"], list)

    def test_full_profile_round_trip(self):
        original = {
            "sessions": 5,
            "questions_asked": 12,
            "last_level": "intermediate",
            "topics_seen": ["transformers", "backprop"],
            "level_history": ["beginner", "intermediate", "intermediate"],
            "topic_counts": {"transformers": 4, "backprop": 8},
            "weak_areas": {"backprop": ["chain rule"]},
            "mastery": {"transformers": 0.7, "backprop": 0.4},
            "used_explanations": {"transformers": ["intermediate-advance"]},
            "recommended_next_topics": ["backprop"],
            "last_evaluation": {"understanding_level": "partial"},
        }
        p = ensure_profile_structure(original)
        self.assertEqual(p["sessions"], 5)
        self.assertEqual(p["topic_counts"]["backprop"], 8)
        self.assertEqual(p["mastery"]["transformers"], 0.7)
        self.assertIn("chain rule", p["weak_areas"]["backprop"])
        self.assertIn("intermediate-advance", p["used_explanations"]["transformers"])


if __name__ == "__main__":
    unittest.main()
