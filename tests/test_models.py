import os
import unittest
from unittest import mock

from langchain_openai import ChatOpenAI

import models


def settings(**values):
    """Run with exactly these settings, whatever the real environment or .env says."""
    managed = ("EMBEDDING_BACKEND", "CHAT_BACKEND", "OPENAI_API_KEY")
    environ = {key: value for key, value in os.environ.items() if key not in managed}
    environ.update(values)
    return mock.patch.dict(os.environ, environ, clear=True)


class BackendTest(unittest.TestCase):
    def test_defaults_to_openai(self):
        with settings():
            self.assertEqual(models.backend("CHAT_BACKEND"), "openai")

    def test_is_case_insensitive_and_ignores_spaces(self):
        with settings(CHAT_BACKEND=" Ollama "):
            self.assertEqual(models.backend("CHAT_BACKEND"), "ollama")

    def test_rejects_an_unknown_backend(self):
        with settings(CHAT_BACKEND="olama"):
            with self.assertRaisesRegex(ValueError, "openai or ollama"):
                models.backend("CHAT_BACKEND")


class CollectionNameTest(unittest.TestCase):
    def test_each_embedding_backend_gets_its_own_collection(self):
        with settings(EMBEDDING_BACKEND="openai"):
            openai_collection = models.collection_name()
        with settings(EMBEDDING_BACKEND="ollama"):
            ollama_collection = models.collection_name()
        self.assertEqual(openai_collection, "text-embedding-3-small")
        self.assertEqual(ollama_collection, "nomic-embed-text")


class MakeModelsTest(unittest.TestCase):
    def test_openai_chat_model_is_deterministic(self):
        with settings(CHAT_BACKEND="openai", OPENAI_API_KEY="test"):
            model = models.make_chat_model()
        self.assertIsInstance(model, ChatOpenAI)
        self.assertEqual(model.model_name, "gpt-4o-mini")
        self.assertEqual(model.temperature, 0)

    def test_ollama_chat_model_sets_the_context_window(self):
        with settings(CHAT_BACKEND="ollama"), mock.patch.object(models, "ChatOllama") as chat:
            models.make_chat_model()
        chat.assert_called_once_with(
            model="llama3.2:3b", temperature=0, num_ctx=8192, validate_model_on_init=True
        )

    def test_stored_text_gets_the_document_prefix(self):
        embeddings = models.NomicEmbeddings(model="nomic-embed-text")
        with mock.patch.object(
            models.OllamaEmbeddings, "embed_documents", return_value=[[0.0]]
        ) as parent:
            embeddings.embed_documents(["Use a checkpointer."])
        parent.assert_called_once_with(["search_document: Use a checkpointer."])

    def test_questions_get_the_query_prefix(self):
        embeddings = models.NomicEmbeddings(model="nomic-embed-text")
        with mock.patch.object(
            models.OllamaEmbeddings, "embed_query", return_value=[0.0]
        ) as parent:
            embeddings.embed_query("What is a checkpointer?")
        parent.assert_called_once_with("search_query: What is a checkpointer?")


class SetupProblemTest(unittest.TestCase):
    def test_missing_key_is_reported_when_openai_is_used(self):
        with settings():
            self.assertIn("OPENAI_API_KEY", models.setup_problem())

    def test_no_key_needed_when_everything_runs_locally(self):
        with settings(EMBEDDING_BACKEND="ollama", CHAT_BACKEND="ollama"), \
             mock.patch.object(models, "NomicEmbeddings"), \
             mock.patch.object(models, "ChatOllama"):
            self.assertIsNone(models.setup_problem())

    def test_ollama_not_running_is_reported(self):
        failure = ValueError("Failed to connect to Ollama.")
        with settings(EMBEDDING_BACKEND="ollama", OPENAI_API_KEY="test"), \
             mock.patch.object(models, "NomicEmbeddings", side_effect=failure):
            self.assertIn("Failed to connect to Ollama", models.setup_problem())

    def test_ingest_does_not_need_the_chat_model(self):
        failure = ValueError("Model `llama3.2:3b` not found in Ollama.")
        with settings(EMBEDDING_BACKEND="openai", CHAT_BACKEND="ollama", OPENAI_API_KEY="test"), \
             mock.patch.object(models, "ChatOllama", side_effect=failure):
            self.assertIsNone(models.setup_problem(include_chat=False))

    def test_unknown_backend_is_reported_not_raised(self):
        with settings(CHAT_BACKEND="gpt", OPENAI_API_KEY="test"):
            self.assertIn("openai or ollama", models.setup_problem())


if __name__ == "__main__":
    unittest.main()
