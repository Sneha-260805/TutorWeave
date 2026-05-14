import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.memory_agent import (
    ensure_profile_structure,
    update_profile_after_question,
    update_profile_after_evaluation,
    record_used_explanation,
    build_memory_hint,
    MASTERY_INITIAL_SCORE,
)


class TestUpdateAfterQuestion(unittest.TestCase):
    def setUp(self):
        self.profile = ensure_profile_structure({})

    def test_increments_questions_asked(self):
        p = update_profile_after_question(self.profile, "neural networks", "beginner")
        self.assertEqual(p["questions_asked"], 1)

    def test_adds_new_topic_to_seen(self):
        p = update_profile_after_question(self.profile, "transformers", "intermediate")
        self.assertIn("transformers", p["topics_seen"])

    def test_duplicate_topic_not_added_twice(self):
        p = update_profile_after_question(self.profile, "backprop", "advanced")
        p = update_profile_after_question(p, "backprop", "advanced")
        self.assertEqual(p["topics_seen"].count("backprop"), 1)

    def test_topic_count_increments_each_visit(self):
        p = update_profile_after_question(self.profile, "backprop", "advanced")
        p = update_profile_after_question(p, "backprop", "advanced")
        self.assertEqual(p["topic_counts"]["backprop"], 2)

    def test_level_recorded_in_history(self):
        p = update_profile_after_question(self.profile, "topic", "advanced")
        self.assertIn("advanced", p["level_history"])


class TestMasteryScoring(unittest.TestCase):
    def _eval(self, profile, topic, level):
        return update_profile_after_evaluation(
            profile, topic, {"understanding_level": level, "weak_concepts": []}
        )

    def test_good_increases_mastery(self):
        p = ensure_profile_structure({"mastery": {"topic": 0.5}})
        p = self._eval(p, "topic", "good")
        self.assertGreater(p["mastery"]["topic"], 0.5)

    def test_partial_decreases_mastery(self):
        p = ensure_profile_structure({"mastery": {"topic": 0.5}})
        p = self._eval(p, "topic", "partial")
        self.assertLess(p["mastery"]["topic"], 0.5)

    def test_poor_decreases_mastery_more_than_partial(self):
        p1 = ensure_profile_structure({"mastery": {"topic": 0.5}})
        p1 = self._eval(p1, "topic", "partial")
        p2 = ensure_profile_structure({"mastery": {"topic": 0.5}})
        p2 = self._eval(p2, "topic", "poor")
        self.assertGreater(p1["mastery"]["topic"], p2["mastery"]["topic"])

    def test_mastery_never_exceeds_one(self):
        p = ensure_profile_structure({"mastery": {"topic": 0.99}})
        for _ in range(20):
            p = self._eval(p, "topic", "good")
        self.assertLessEqual(p["mastery"]["topic"], 1.0)

    def test_mastery_never_goes_below_zero(self):
        p = ensure_profile_structure({"mastery": {"topic": 0.01}})
        for _ in range(20):
            p = self._eval(p, "topic", "poor")
        self.assertGreaterEqual(p["mastery"]["topic"], 0.0)

    def test_diminishing_returns_on_good(self):
        # Starting from low mastery should give a bigger absolute gain than from high
        p_low = ensure_profile_structure({"mastery": {"topic": 0.2}})
        p_low = self._eval(p_low, "topic", "good")
        gain_from_low = p_low["mastery"]["topic"] - 0.2

        p_high = ensure_profile_structure({"mastery": {"topic": 0.9}})
        p_high = self._eval(p_high, "topic", "good")
        gain_from_high = p_high["mastery"]["topic"] - 0.9

        self.assertGreater(gain_from_low, gain_from_high)

    def test_new_topic_initialised_at_default(self):
        p = ensure_profile_structure({})
        p = self._eval(p, "new_topic", "good")
        self.assertIn("new_topic", p["mastery"])

    def test_weak_concepts_recorded(self):
        p = ensure_profile_structure({})
        p = update_profile_after_evaluation(
            p, "backprop", {"understanding_level": "poor", "weak_concepts": ["chain rule", "gradient flow"]}
        )
        self.assertIn("chain rule", p["weak_areas"]["backprop"])
        self.assertIn("gradient flow", p["weak_areas"]["backprop"])

    def test_good_with_no_weak_concepts_clears_all(self):
        p = ensure_profile_structure({})
        p = update_profile_after_evaluation(
            p, "backprop", {"understanding_level": "poor", "weak_concepts": ["chain rule", "gradient flow"]}
        )
        p = update_profile_after_evaluation(
            p, "backprop", {"understanding_level": "good", "weak_concepts": []}
        )
        self.assertEqual(p["weak_areas"]["backprop"], [])

    def test_good_with_specific_concepts_removes_only_those(self):
        p = ensure_profile_structure({})
        p = update_profile_after_evaluation(
            p, "backprop", {"understanding_level": "poor", "weak_concepts": ["chain rule", "gradient flow"]}
        )
        p = update_profile_after_evaluation(
            p, "backprop", {"understanding_level": "good", "weak_concepts": ["chain rule"]}
        )
        self.assertNotIn("chain rule", p["weak_areas"]["backprop"])
        self.assertIn("gradient flow", p["weak_areas"]["backprop"])


class TestUsedExplanations(unittest.TestCase):
    def test_records_explanation_tag(self):
        p = ensure_profile_structure({})
        p = record_used_explanation(p, "neural networks", "beginner-default")
        self.assertIn("beginner-default", p["used_explanations"]["neural networks"])

    def test_no_duplicates_recorded(self):
        p = ensure_profile_structure({})
        p = record_used_explanation(p, "topic", "beginner-default")
        p = record_used_explanation(p, "topic", "beginner-default")
        self.assertEqual(len(p["used_explanations"]["topic"]), 1)

    def test_multiple_tags_per_topic(self):
        p = ensure_profile_structure({})
        p = record_used_explanation(p, "topic", "beginner-default")
        p = record_used_explanation(p, "topic", "intermediate-advance")
        self.assertEqual(len(p["used_explanations"]["topic"]), 2)

    def test_memory_hint_includes_used_explanations(self):
        p = ensure_profile_structure({})
        p = record_used_explanation(p, "topic", "beginner-remedial")
        hint = build_memory_hint(p, "topic")
        self.assertIn("beginner-remedial", hint)

    def test_separate_topics_isolated(self):
        p = ensure_profile_structure({})
        p = record_used_explanation(p, "topic_a", "beginner-default")
        self.assertNotIn("topic_b", p["used_explanations"])


class TestMemoryHint(unittest.TestCase):
    def test_no_history_gives_default_hint(self):
        p = ensure_profile_structure({})
        hint = build_memory_hint(p, "some_topic")
        self.assertIsInstance(hint, str)
        self.assertGreater(len(hint), 0)

    def test_weak_areas_mentioned_in_hint(self):
        p = ensure_profile_structure({"weak_areas": {"ml": ["gradient descent"]}})
        p["topic_counts"]["ml"] = 1
        hint = build_memory_hint(p, "ml")
        self.assertIn("gradient descent", hint)

    def test_high_mastery_suggests_deeper_content(self):
        p = ensure_profile_structure({"mastery": {"ml": 0.9}})
        hint = build_memory_hint(p, "ml")
        self.assertIn("deeper", hint.lower())

    def test_low_mastery_suggests_simpler_content(self):
        p = ensure_profile_structure({"mastery": {"ml": 0.2}})
        hint = build_memory_hint(p, "ml")
        self.assertIn("simpler", hint.lower())


if __name__ == "__main__":
    unittest.main()
