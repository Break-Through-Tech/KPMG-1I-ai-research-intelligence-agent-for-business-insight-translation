"""Agent tests. Qdrant and the LLM are faked."""

import threading
from dataclasses import asdict

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from src import agent
from src.agent import NO_EVIDENCE_ANSWER, GenerationError, build_graph, format_context, run_agent
from src.prompts import answer_prompt
from src.retrieval import ChunkHit, PaperHit, RetrievalError

QUESTION = "How does 4-bit quantization affect LLM accuracy?"


def paper(rank=1, arxiv_id="2610.13767v1", title="DualQuant", abstract="We quantize LLMs to 4 bits.") -> dict:
    return asdict(PaperHit(arxiv_id=arxiv_id, title=title, abstract=abstract, score=0.8, rank=rank))


def chunk(rank=1, arxiv_id="2610.29544v1", title="RISK-KV", content="Selective KV reconstruction.",
          breadcrumb=("3 Method", "3.2 Selection"), content_type="text") -> dict:
    return asdict(ChunkHit(chunk_id=f"{arxiv_id}_chunk_{rank}", arxiv_id=arxiv_id, title=title, content=content,
                           breadcrumb=list(breadcrumb), chunk_index=rank, content_type=content_type,
                           score=0.03, rank=rank))


class TestFormatContext:
    def test_paper_shows_rank_id_title_and_abstract(self):
        context = format_context([paper()], [chunk()])
        assert "[P1] | arXiv:2610.13767v1 | DualQuant\nAbstract: We quantize LLMs to 4 bits." in context

    def test_passage_keeps_its_paper_section_and_type(self):
        context = format_context([paper()], [chunk()])
        assert ("[C1] | arXiv:2610.29544v1 | RISK-KV | Section: 3 Method > 3.2 Selection | Type: text\n"
                "Selective KV reconstruction.") in context

    def test_papers_and_passages_are_separate_sections_in_search_order(self):
        context = format_context([paper(1), paper(2, "2610.12338v1", "VFold")],
                                 [chunk(1), chunk(2, "2610.12327v1", "SparseDecoding")])
        papers, passages = context.split("## Passages")
        assert papers.startswith("## Papers")
        assert papers.index("DualQuant") < papers.index("VFold")
        assert "Selective KV reconstruction." not in papers
        assert "Abstract:" not in passages
        assert passages.index("RISK-KV") < passages.index("SparseDecoding")

    def test_scores_are_left_out(self):
        context = format_context([paper()], [chunk()])
        assert "0.8" not in context and "0.03" not in context

    def test_only_papers(self):
        context = format_context([paper()], [])
        assert "DualQuant" in context
        assert context.endswith("## Passages\n\nNo passages found.")

    def test_only_passages(self):
        context = format_context([], [chunk()])
        assert "## Papers\n\nNo papers found.\n\n## Passages" in context
        assert "RISK-KV" in context

    def test_nothing_found_gives_empty_context(self):
        assert format_context([], []) == ""

    def test_missing_metadata_is_left_out_not_invented(self):
        context = format_context(
            [paper(title=None, abstract=None)],
            [chunk(title=None, breadcrumb=(), content_type=None)],
        )
        assert "[P1] | arXiv:2610.13767v1\n" in context
        assert "[C1] | arXiv:2610.29544v1\nSelective KV reconstruction." in context
        assert "None" not in context and "Abstract" not in context and "Section" not in context


class TestAnswerPrompt:
    def test_question_and_context_reach_the_model(self):
        system, human = answer_prompt.format_messages(question="How does 4-bit quantization work?",
                                                      context="## Papers\n\n[P1] | arXiv:2610.13767v1")
        assert "[Paper Title](https://arxiv.org/abs/ARXIV_ID)" in system.content
        assert human.content.endswith("Question: How does 4-bit quantization work?")
        assert "[P1] | arXiv:2610.13767v1" in human.content


PAPER_HIT = PaperHit(arxiv_id="2610.13767v1", title="DualQuant", abstract="We quantize LLMs to 4 bits.", score=0.8, rank=1)
CHUNK_HIT = ChunkHit(chunk_id="2610.29544v1_chunk_3", arxiv_id="2610.29544v1", title="RISK-KV",
                     content="Selective KV reconstruction.", breadcrumb=["3 Method"], chunk_index=3,
                     content_type="text", score=0.03, rank=1)
CLIENT = object()


class FakeSearch:
    """Stands in for search_papers / search_chunks and records each call."""

    def __init__(self, hits=(), error=None, barrier=None):
        self.hits, self.error, self.barrier, self.calls = list(hits), error, barrier, []

    def __call__(self, client, question):
        self.calls.append((client, question))
        if self.barrier:
            self.barrier.wait()
        if self.error:
            raise self.error
        return self.hits


class FakeLLM:
    """A runnable LLM that records the prompt it receives."""

    def __init__(self, answer="DualQuant reaches 4 bits [DualQuant](https://arxiv.org/abs/2610.13767v1).", error=None):
        self.answer, self.error, self.prompts = answer, error, []
        self.runnable = RunnableLambda(self._reply)

    def _reply(self, prompt):
        self.prompts.append(prompt.to_messages())
        if self.error:
            raise self.error
        return AIMessage(content=self.answer)


@pytest.fixture
def searches(monkeypatch):
    papers, chunks = FakeSearch([PAPER_HIT]), FakeSearch([CHUNK_HIT])
    monkeypatch.setattr(agent, "search_papers", papers)
    monkeypatch.setattr(agent, "search_chunks", chunks)
    return papers, chunks


def run_graph(llm=None):
    return build_graph(CLIENT, (llm or FakeLLM()).runnable).invoke({"query": QUESTION})


class TestGraph:
    def test_end_to_end_returns_hits_context_and_answer(self, searches):
        state = run_graph()
        assert state["query"] == QUESTION
        assert state["paper_hits"] == [asdict(PAPER_HIT)]
        assert state["chunk_hits"] == [asdict(CHUNK_HIT)]
        assert "DualQuant" in state["context"] and "Selective KV reconstruction." in state["context"]
        assert state["answer"].startswith("DualQuant reaches 4 bits")

    def test_both_searches_get_the_original_query_and_the_shared_client(self, searches):
        run_graph()
        papers, chunks = searches
        assert papers.calls == [(CLIENT, QUESTION)]
        assert chunks.calls == [(CLIENT, QUESTION)]

    def test_searches_run_in_parallel(self, monkeypatch):
        # Each search waits until the other has started; run one after the other, they'd time out.
        barrier = threading.Barrier(2, timeout=5)
        monkeypatch.setattr(agent, "search_papers", FakeSearch([PAPER_HIT], barrier=barrier))
        monkeypatch.setattr(agent, "search_chunks", FakeSearch([CHUNK_HIT], barrier=barrier))
        assert run_graph()["answer"]

    def test_context_is_assembled_once_with_both_results(self, searches, monkeypatch):
        seen = []
        monkeypatch.setattr(agent, "format_context", lambda papers, chunks: seen.append((papers, chunks)) or "ctx")
        run_graph()
        assert seen == [([asdict(PAPER_HIT)], [asdict(CHUNK_HIT)])]

    def test_llm_gets_the_original_query_and_the_assembled_context(self, searches):
        llm = FakeLLM()
        state = run_graph(llm)
        [[system, human]] = llm.prompts
        assert "research assistant" in system.content
        assert human.content == f"Retrieved context:\n\n{state['context']}\n\nQuestion: {QUESTION}"

    def test_no_papers_still_answers_from_passages(self, searches):
        searches[0].hits = []
        llm = FakeLLM()
        state = run_graph(llm)
        assert "No papers found." in state["context"] and "RISK-KV" in state["context"]
        assert len(llm.prompts) == 1 and state["answer"]

    def test_no_passages_still_answers_from_papers(self, searches):
        searches[1].hits = []
        state = run_graph()
        assert "No passages found." in state["context"] and "DualQuant" in state["context"]

    def test_nothing_found_skips_the_llm(self, searches):
        searches[0].hits, searches[1].hits = [], []
        llm = FakeLLM()
        state = run_graph(llm)
        assert state["context"] == "" and state["answer"] == NO_EVIDENCE_ANSWER
        assert llm.prompts == []

    @pytest.mark.parametrize("failing", [0, 1])
    def test_retrieval_failure_propagates(self, searches, failing):
        searches[failing].error = RetrievalError("Couldn't reach Qdrant: connection refused")
        llm = FakeLLM()
        with pytest.raises(RetrievalError, match="connection refused"):
            run_graph(llm)
        assert llm.prompts == []

    def test_llm_failure_raises_generation_error_with_the_cause(self, searches):
        with pytest.raises(GenerationError, match="Ollama is not running") as error:
            run_graph(FakeLLM(error=ConnectionError("Ollama is not running")))
        assert isinstance(error.value.__cause__, ConnectionError)

    def test_empty_llm_answer_is_an_error(self, searches):
        with pytest.raises(GenerationError, match="empty answer"):
            run_graph(FakeLLM(answer="   "))


class FakeClient:
    closed = False

    def close(self):
        self.closed = True


class TestRunAgent:
    @pytest.mark.parametrize("query", ["", "   "])
    def test_blank_query_is_rejected_before_opening_qdrant(self, monkeypatch, query):
        monkeypatch.setattr(agent, "get_client", lambda: pytest.fail("opened Qdrant for a blank query"))
        with pytest.raises(ValueError, match="blank"):
            run_agent(query)

    def test_runs_the_stripped_query_and_closes_the_client(self, searches, monkeypatch):
        client = FakeClient()
        monkeypatch.setattr(agent, "get_client", lambda: client)
        monkeypatch.setattr(agent, "get_llm", lambda: FakeLLM().runnable)
        state = run_agent(f"  {QUESTION}  ")
        assert searches[0].calls == [(client, QUESTION)]
        assert state["answer"] and client.closed

    def test_client_is_closed_when_retrieval_fails(self, searches, monkeypatch):
        client = FakeClient()
        monkeypatch.setattr(agent, "get_client", lambda: client)
        monkeypatch.setattr(agent, "get_llm", lambda: FakeLLM().runnable)
        searches[1].error = RetrievalError("Chunk search failed")
        with pytest.raises(RetrievalError):
            run_agent(QUESTION)
        assert client.closed
