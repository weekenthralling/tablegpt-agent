import sys
import unittest
from unittest.mock import MagicMock, patch


class TestTableGPTInit(unittest.TestCase):
    def setUp(self):
        # Save the original tablegpt module if it exists
        self.original_tablegpt = sys.modules.get("tablegpt")

        # Clear tablegpt from sys.modules
        if "tablegpt" in sys.modules:
            del sys.modules["tablegpt"]

        # Create a mock site module
        self.mock_site = MagicMock()
        self.original_site = sys.modules["site"]
        sys.modules["site"] = self.mock_site

    def tearDown(self):
        # Restore the original modules
        sys.modules["site"] = self.original_site

        # Restore the original tablegpt module if it existed
        if self.original_tablegpt:
            sys.modules["tablegpt"] = self.original_tablegpt
        elif "tablegpt" in sys.modules:
            del sys.modules["tablegpt"]

    def test_find_tablegpt_ipykernel_profile_dir_found(self):
        # mock return values
        self.mock_site.getsitepackages.return_value = ["/usr/local/lib/python3.x/site-packages"]
        self.mock_site.getusersitepackages.return_value = ""

        with patch("pathlib.Path.glob", return_value=iter(["mock-udfs.py"])):
            from tablegpt import DEFAULT_TABLEGPT_IPYKERNEL_PROFILE_DIR

            assert DEFAULT_TABLEGPT_IPYKERNEL_PROFILE_DIR == "/usr/local/share/ipykernel/profile/tablegpt"

    def test_default_tablegpt_ipykernel_profile_dir_found_second_path(self):
        # mock return values
        self.mock_site.getsitepackages.return_value = [
            "/wrong/lib/python3.x/site-packages",
            "/usr/local/lib/python3.x/site-packages",
        ]
        self.mock_site.getusersitepackages.return_value = "/home/user/.local/lib/python3.x/site-packages"

        # Found in the second possible path.
        with patch("pathlib.Path.glob", side_effect=[iter([]), iter(["mock-udfs.py"]), iter([])]):
            from tablegpt import DEFAULT_TABLEGPT_IPYKERNEL_PROFILE_DIR

            assert DEFAULT_TABLEGPT_IPYKERNEL_PROFILE_DIR == "/usr/local/share/ipykernel/profile/tablegpt"

    def test_default_tablegpt_ipykernel_profile_dir_not_found(self):
        # mock return values
        self.mock_site.getsitepackages.return_value = [
            "/wrong/lib/python3.x/site-packages",
            "/other/lib/python3.x/site-packages",
        ]
        self.mock_site.getusersitepackages.return_value = "/home/user/.local/lib/python3.x/site-packages"

        # not found
        with patch("pathlib.Path.glob", side_effect=[iter([]), iter([]), iter([])]), self.assertWarns(UserWarning):
            from tablegpt import DEFAULT_TABLEGPT_IPYKERNEL_PROFILE_DIR

            assert DEFAULT_TABLEGPT_IPYKERNEL_PROFILE_DIR is None


if __name__ == "__main__":
    unittest.main()
