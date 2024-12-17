import contextlib
import unittest
from unittest.mock import patch

from tablegpt import _find_tablegpt_ipykernel_profile_dir, possible_installation_locations


class TestTableGPTInit(unittest.TestCase):
    @patch("tablegpt.Path.glob")
    def test_find_tablegpt_ipykernel_profile_dir_found(self, mock_glob):
        mock_glob.return_value = iter(["mock-udfs.py"])
        result = _find_tablegpt_ipykernel_profile_dir("/mock/path/to/site-packages")
        assert result == "/mock/share/ipykernel/profile/tablegpt"

    @patch("tablegpt.Path.glob")
    def test_default_tablegpt_ipykernel_profile_dir_found(self, mock_glob):
        mock_glob.side_effect = [iter([]), iter(["mock-udfs.py"]), iter([])]
        with (
            patch("sys.path", ["/wrong/path/to/site-packages", "/mock/path/to/site-packages", "/another/wrong/path"]),
            contextlib.suppress(StopIteration),
        ):
            result = next(filter(_find_tablegpt_ipykernel_profile_dir, possible_installation_locations))
            assert result == "/mock/share/ipykernel/profile/tablegpt"

    @patch("tablegpt.Path.glob")
    def test_default_tablegpt_ipykernel_profile_dir_not_found(self, mock_glob):
        mock_glob.side_effect = [iter([]), iter([]), iter([])]
        with (
            patch("sys.path", ["/wrong/path/to/site-packages", "/another/wrong/path"]),
            contextlib.suppress(StopIteration),
        ):
            result = next(filter(_find_tablegpt_ipykernel_profile_dir, possible_installation_locations))
            assert result is None


if __name__ == "__main__":
    unittest.main()
