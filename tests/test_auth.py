import sys
import os
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auth.password_utils import hash_password, verify_password
from auth.auth_service import _is_valid_email, register_user


class TestPasswordUtils(unittest.TestCase):
    def test_correct_password_verifies(self):
        h = hash_password("correcthorse")
        self.assertTrue(verify_password("correcthorse", h))

    def test_wrong_password_rejected(self):
        h = hash_password("correct")
        self.assertFalse(verify_password("wrong", h))

    def test_each_hash_is_unique(self):
        pw = "same_password"
        self.assertNotEqual(hash_password(pw), hash_password(pw))

    def test_truncated_hash_rejected(self):
        h = hash_password("test123")
        self.assertFalse(verify_password("test123", h[:20]))

    def test_empty_password_rejected(self):
        h = hash_password("real")
        self.assertFalse(verify_password("", h))


class TestEmailValidation(unittest.TestCase):
    def test_standard_email(self):
        self.assertTrue(_is_valid_email("user@example.com"))

    def test_subdomain_email(self):
        self.assertTrue(_is_valid_email("user.name@mail.domain.co.uk"))

    def test_missing_at_sign(self):
        self.assertFalse(_is_valid_email("notanemail"))

    def test_missing_local_part(self):
        self.assertFalse(_is_valid_email("@domain.com"))

    def test_missing_domain(self):
        self.assertFalse(_is_valid_email("user@"))

    def test_no_dot_in_domain(self):
        self.assertFalse(_is_valid_email("user@domain"))

    def test_empty_string(self):
        self.assertFalse(_is_valid_email(""))


class TestRegisterUser(unittest.TestCase):
    def _register(self, name="Alice", username=None, email="alice@example.com", password="secure123"):
        with patch("auth.auth_service.create_user", return_value=1):
            return register_user(name, username, email, password)

    def test_valid_registration_succeeds(self):
        ok, msg = self._register()
        self.assertTrue(ok)

    def test_missing_name_fails(self):
        ok, msg = self._register(name="")
        self.assertFalse(ok)

    def test_invalid_email_fails(self):
        ok, msg = self._register(email="notanemail")
        self.assertFalse(ok)
        self.assertIn("email", msg.lower())

    def test_short_password_fails(self):
        ok, msg = self._register(password="12345")
        self.assertFalse(ok)
        self.assertIn("6", msg)

    def test_duplicate_user_fails(self):
        with patch("auth.auth_service.create_user", return_value=None):
            ok, msg = register_user("Bob", None, "bob@example.com", "password123")
        self.assertFalse(ok)
        self.assertIn("already exists", msg.lower())


if __name__ == "__main__":
    unittest.main()
