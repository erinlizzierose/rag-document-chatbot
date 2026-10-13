import unittest

from langchain_core.documents import Document

import app


class DescribeSourceTest(unittest.TestCase):
    def test_pdf_pages_are_shown_counting_from_one(self):
        doc = Document("x", metadata={"source": "data/notes.pdf", "page": 0})
        self.assertEqual(app.describe_source(doc), "data/notes.pdf, page 1")

    def test_text_files_show_just_the_path(self):
        doc = Document("x", metadata={"source": "data/notes.md"})
        self.assertEqual(app.describe_source(doc), "data/notes.md")

    def test_missing_source_has_a_fallback(self):
        self.assertEqual(app.describe_source(Document("x")), "unknown source")


if __name__ == "__main__":
    unittest.main()
