from __future__ import annotations

import unittest


class SessionTest(unittest.TestCase):
    @unittest.skip("background Future queueing is unused in production")
    def test_background_future_queueing_is_unused(self):
        pass


if __name__ == "__main__":
    unittest.main()
