import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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


class IngestMainTest(unittest.TestCase):
    def run_main_with(self, files, problem=None):
        """Run ingest.main() against a temporary data folder, with Chroma and the models mocked out."""
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            data_dir.mkdir()
            for name, text in files.items():
                (data_dir / name).write_text(text)
            with mock.patch.object(ingest, "DATA_DIR", data_dir), \
                 mock.patch.object(ingest, "VECTORSTORE_DIR", Path(tmp) / "vectorstore"), \
                 mock.patch.object(ingest, "Chroma") as chroma, \
                 mock.patch.object(ingest.models, "make_embeddings"), \
                 mock.patch.object(ingest.models, "setup_problem", return_value=problem):
                # Kept on self so a test can inspect it even when main() raises.
                self.chroma = chroma
                # main() prints progress lines; keep them out of the test output.
                with contextlib.redirect_stdout(io.StringIO()):
                    ingest.main()
            return chroma

    def test_a_setup_problem_stops_before_touching_the_index(self):
        with self.assertRaisesRegex(SystemExit, "No OPENAI_API_KEY"):
            self.run_main_with({"real.md": "Text."}, problem="No OPENAI_API_KEY found.")
        self.chroma.assert_not_called()

    def test_each_embedding_model_writes_its_own_collection(self):
        with mock.patch.dict(os.environ, {"EMBEDDING_BACKEND": "ollama"}):
            chroma = self.run_main_with({"real.md": "# Memory\nUse a checkpointer."})
        self.assertEqual(chroma.call_args.kwargs["collection_name"], "nomic-embed-text")

    def test_stops_before_touching_the_index_when_no_text_is_found(self):
        with self.assertRaises(SystemExit):
            self.run_main_with({"scanned.md": "", "blank.txt": "   \n"})
        self.chroma.assert_not_called()

    def test_indexes_when_at_least_one_file_has_text(self):
        chroma = self.run_main_with({"empty.md": "", "real.md": "# Memory\nUse a checkpointer."})
        store = chroma.return_value
        store.reset_collection.assert_called_once()
        chunks = store.add_documents.call_args.args[0]
        self.assertEqual(len(chunks), 1)
        self.assertIn("checkpointer", chunks[0].page_content)


if __name__ == "__main__":
    unittest.main()
