import unittest

import ingest


class StripLeadingBlockquoteTest(unittest.TestCase):
    def test_removes_the_blockquote_at_the_top(self):
        text = "> Note for LLMs\n> See llms.txt\n\n# Title\nBody"
        self.assertEqual(ingest.strip_leading_blockquote(text), "# Title\nBody")

    def test_keeps_blockquotes_further_down(self):
        text = "# Title\n\n> A real quote"
        self.assertEqual(ingest.strip_leading_blockquote(text), text)

    def test_empty_text_stays_empty(self):
        self.assertEqual(ingest.strip_leading_blockquote(""), "")


if __name__ == "__main__":
    unittest.main()
