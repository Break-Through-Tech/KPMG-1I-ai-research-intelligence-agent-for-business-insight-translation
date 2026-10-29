import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from src import agent


class TestAgentCli(unittest.TestCase):
    def test_interactive_mode_reuses_run_agent_for_multiple_queries(self):
        first = {
            "answer": "first answer",
            "citations": [],
            "timings": {"total_execution_seconds": 0.1},
        }
        second = {
            "answer": "second answer",
            "citations": [],
            "timings": {"total_execution_seconds": 0.2},
        }

        with patch("builtins.input", side_effect=["first question", "second question", ""]), \
                patch("src.agent.run_agent", side_effect=[first, second]) as run_agent:
            agent.run_interactive()

        self.assertEqual(run_agent.call_args_list[0].args, ("first question",))
        self.assertEqual(run_agent.call_args_list[1].args, ("second question",))

    def test_timing_output_is_separate_from_answer(self):
        result = {
            "answer": "answer text",
            "citations": [],
            "timings": {"total_execution_seconds": 0.1},
        }
        output = io.StringIO()

        with redirect_stdout(output):
            agent.print_result(result)

        rendered = output.getvalue()
        self.assertIn("Answer:\nanswer text", rendered)
        self.assertIn("Timing:\n- total_execution_seconds: 0.100s", rendered)
        self.assertEqual(result["answer"], "answer text")

    def test_ollama_keep_alive_uses_environment_override(self):
        with patch("src.agent.OLLAMA_KEEP_ALIVE", "25m"), \
                patch("src.agent.ChatOllama") as chat_ollama:
            agent.create_llm()

        self.assertEqual(chat_ollama.call_args.kwargs["keep_alive"], "25m")