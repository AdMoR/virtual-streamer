"""
Tests for MapReduceAgent architecture.

Tests cover:
- MapperAgent abstract base class
- AggregatorAgent abstract base class
- MapReduceAgent orchestration
- SentenceVideoMapper concrete implementation
- SentenceVideoAggregator concrete implementation
"""

import pytest
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from virtual_streamer.lib.agents import (
    MapperAgent,
    AggregatorAgent,
    MapReduceAgent,
    StatefulWorker,
)
from virtual_streamer.agents.sentence_video_matcher.agent import (
    SentenceVideoMapper,
    SentenceVideoAggregator,
    SentenceVideoMatcherAgent,
    create_sentence_video_matcher,
    SENTENCES_KEY,
    MATCHES_KEY,
)
from virtual_streamer.agents.video_matcher.schema import (
    VideoMatchResult,
    ContextualRating,
)
from virtual_streamer.agents.sentence_video_matcher.schema import (
    SentenceVideoMatcherOutput,
)


# ============================================================================
# Test Schemas and Mocks
# ============================================================================


class SimpleInput(BaseModel):
    """Simple input schema for testing."""
    value: str = Field(description="A simple value")


class SimpleOutput(BaseModel):
    """Simple output schema for testing."""
    result: str = Field(description="A simple result")
    score: int = Field(description="A score", ge=0, le=10)


class MockVideoRetriever:
    """Mock video retriever for testing."""
    
    def __init__(self, video_map: Optional[Dict[str, List[str]]] = None):
        self.video_map = video_map or {}
        self.default_videos = ["/videos/default1.mp4", "/videos/default2.mp4"]
        self.search_calls: List[tuple] = []
    
    def search(self, query: str, top_k: int = 10) -> List[str]:
        self.search_calls.append((query, top_k))
        return self.video_map.get(query, self.default_videos)[:top_k]


class MockStatefulWorker:
    """Mock worker that implements StatefulWorker protocol."""
    
    def __init__(self, run_id: str, input_schema=SimpleInput, output_schema=SimpleOutput):
        self.run_id = run_id
        self._input_schema = input_schema
        self._output_schema = output_schema
        self.name = f"mock_worker_{run_id}"
    
    def get_input_key(self) -> str:
        return f"task:{self.run_id}:input"
    
    def get_input_schema(self):
        return self._input_schema
    
    def get_output_key(self) -> str:
        return f"result:{self.run_id}:output"
    
    def get_output_schema(self):
        return self._output_schema


class MockSessionState:
    """Mock session state for testing."""
    
    def __init__(self, initial_state: dict = None):
        self._state = initial_state or {}
    
    def get(self, key, default=None):
        return self._state.get(key, default)
    
    def __setitem__(self, key, value):
        self._state[key] = value
    
    def __getitem__(self, key):
        return self._state[key]
    
    def __contains__(self, key):
        return key in self._state
    
    def items(self):
        return self._state.items()
    
    def keys(self):
        return self._state.keys()


class MockSession:
    """Mock session for testing."""
    
    def __init__(self, initial_state: dict = None):
        self.state = MockSessionState(initial_state)


class MockInvocationContext:
    """Mock invocation context for testing build_items_from_state and aggregation_fn."""
    
    def __init__(self, initial_state: dict = None):
        self.session = MockSession(initial_state)


# ============================================================================
# MapperAgent Tests
# ============================================================================


class TestMapperAgentAbstract:
    """Test MapperAgent abstract base class."""
    
    def test_mapper_is_abstract(self):
        """Test that MapperAgent cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract"):
            MapperAgent(worker_factory=lambda x: None)
    
    def test_concrete_mapper_requires_build_items_from_state(self):
        """Test that subclass must implement build_items_from_state."""
        class IncompleteMapper(MapperAgent):
            pass
        
        with pytest.raises(TypeError, match="abstract"):
            IncompleteMapper(worker_factory=lambda x: None)
    
    def test_concrete_mapper_can_be_instantiated(self):
        """Test that a properly implemented mapper can be created."""
        class ConcreteMapper(MapperAgent):
            def build_items_from_state(self, ctx):
                return [{"value": "test"}]
        
        mapper = ConcreteMapper(worker_factory=lambda x: MockStatefulWorker(x))
        assert mapper is not None
        assert mapper.name == "mapper"
    
    def test_mapper_custom_name(self):
        """Test mapper with custom name."""
        class ConcreteMapper(MapperAgent):
            def build_items_from_state(self, ctx):
                return []
        
        mapper = ConcreteMapper(
            worker_factory=lambda x: MockStatefulWorker(x),
            name="custom_mapper"
        )
        assert mapper.name == "custom_mapper"
    
    def test_get_output_keys_empty_before_run(self):
        """Test that get_output_keys returns empty before run."""
        class ConcreteMapper(MapperAgent):
            def build_items_from_state(self, ctx):
                return []
        
        mapper = ConcreteMapper(worker_factory=lambda x: MockStatefulWorker(x))
        assert mapper.get_output_keys() == []
    
    def test_get_output_schema_none_before_run(self):
        """Test that get_output_schema returns None before run."""
        class ConcreteMapper(MapperAgent):
            def build_items_from_state(self, ctx):
                return []
        
        mapper = ConcreteMapper(worker_factory=lambda x: MockStatefulWorker(x))
        assert mapper.get_output_schema() is None


class TestMapperAgentBuildItems:
    """Test MapperAgent.build_items_from_state behavior."""
    
    def test_build_items_receives_context(self):
        """Test that build_items_from_state receives the context."""
        received_ctx = []
        
        class TrackingMapper(MapperAgent):
            def build_items_from_state(self, ctx):
                received_ctx.append(ctx)
                return []
        
        ctx = MockInvocationContext({"test_key": "test_value"})
        mapper = TrackingMapper(worker_factory=lambda x: MockStatefulWorker(x))
        
        # Call build_items_from_state directly
        items = mapper.build_items_from_state(ctx)
        
        assert len(received_ctx) == 1
        assert received_ctx[0] is ctx
    
    def test_build_items_can_read_state(self):
        """Test that build_items_from_state can read from state."""
        class StateReadingMapper(MapperAgent):
            def build_items_from_state(self, ctx):
                data = ctx.session.state.get("input_data", [])
                return [{"value": item} for item in data]
        
        ctx = MockInvocationContext({"input_data": ["a", "b", "c"]})
        mapper = StateReadingMapper(worker_factory=lambda x: MockStatefulWorker(x))
        
        items = mapper.build_items_from_state(ctx)
        
        assert len(items) == 3
        assert items[0] == {"value": "a"}
        assert items[1] == {"value": "b"}
        assert items[2] == {"value": "c"}
    
    def test_build_items_returns_correct_format(self):
        """Test that build_items returns list of dicts."""
        class FormatMapper(MapperAgent):
            def build_items_from_state(self, ctx):
                return [
                    {"key1": "value1", "key2": 123},
                    {"key1": "value2", "key2": 456},
                ]
        
        ctx = MockInvocationContext()
        mapper = FormatMapper(worker_factory=lambda x: MockStatefulWorker(x))
        
        items = mapper.build_items_from_state(ctx)
        
        assert isinstance(items, list)
        assert all(isinstance(item, dict) for item in items)


# ============================================================================
# AggregatorAgent Tests
# ============================================================================


class TestAggregatorAgentAbstract:
    """Test AggregatorAgent abstract base class."""
    
    def test_aggregator_is_abstract(self):
        """Test that AggregatorAgent cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract"):
            AggregatorAgent(
                name="test",
                input_keys=[],
                input_schema=SimpleOutput,
                output_key="result",
            )
    
    def test_concrete_aggregator_requires_aggregation_fn(self):
        """Test that subclass must implement aggregation_fn."""
        class IncompleteAggregator(AggregatorAgent):
            pass
        
        with pytest.raises(TypeError, match="abstract"):
            IncompleteAggregator(
                name="test",
                input_keys=[],
                input_schema=SimpleOutput,
                output_key="result",
            )
    
    def test_concrete_aggregator_can_be_instantiated(self):
        """Test that a properly implemented aggregator can be created."""
        class ConcreteAggregator(AggregatorAgent[SimpleOutput]):
            async def aggregation_fn(self, results):
                return results[0] if results else None
        
        agg = ConcreteAggregator(
            name="test_agg",
            input_keys=["key1", "key2"],
            input_schema=SimpleOutput,
            output_key="aggregated_result",
        )
        assert agg is not None
        assert agg.name == "test_agg"
        assert agg._input_keys == ["key1", "key2"]
        assert agg._output_key == "aggregated_result"


class TestAggregatorAgentCollectResults:
    """Test AggregatorAgent._collect_results behavior."""
    
    def test_collect_results_parses_json(self):
        """Test that _collect_results parses JSON strings."""
        class SimpleAggregator(AggregatorAgent[SimpleOutput]):
            async def aggregation_fn(self, results):
                return results[0] if results else None
        
        state = {
            "key1": SimpleOutput(result="test1", score=5).model_dump_json(),
            "key2": SimpleOutput(result="test2", score=8).model_dump_json(),
        }
        ctx = MockInvocationContext(state)
        
        agg = SimpleAggregator(
            name="test",
            input_keys=["key1", "key2"],
            input_schema=SimpleOutput,
            output_key="result",
        )
        
        results = agg._collect_results(ctx)
        
        assert len(results) == 2
        assert results[0].result == "test1"
        assert results[0].score == 5
        assert results[1].result == "test2"
        assert results[1].score == 8
    
    def test_collect_results_parses_dicts(self):
        """Test that _collect_results parses dict values."""
        class SimpleAggregator(AggregatorAgent[SimpleOutput]):
            async def aggregation_fn(self, results):
                return results[0] if results else None
        
        state = {
            "key1": {"result": "dict_test", "score": 7},
        }
        ctx = MockInvocationContext(state)
        
        agg = SimpleAggregator(
            name="test",
            input_keys=["key1"],
            input_schema=SimpleOutput,
            output_key="result",
        )
        
        results = agg._collect_results(ctx)
        
        assert len(results) == 1
        assert results[0].result == "dict_test"
        assert results[0].score == 7
    
    def test_collect_results_skips_missing_keys(self):
        """Test that _collect_results skips missing keys."""
        class SimpleAggregator(AggregatorAgent[SimpleOutput]):
            async def aggregation_fn(self, results):
                return results[0] if results else None
        
        state = {
            "key1": SimpleOutput(result="exists", score=5).model_dump_json(),
            # key2 is missing
        }
        ctx = MockInvocationContext(state)
        
        agg = SimpleAggregator(
            name="test",
            input_keys=["key1", "key2", "key3"],
            input_schema=SimpleOutput,
            output_key="result",
        )
        
        results = agg._collect_results(ctx)
        
        assert len(results) == 1
        assert results[0].result == "exists"
    
    def test_collect_results_only_reads_specified_keys(self):
        """Test that _collect_results only reads from specified keys."""
        class SimpleAggregator(AggregatorAgent[SimpleOutput]):
            async def aggregation_fn(self, results):
                return results[0] if results else None
        
        state = {
            "key1": SimpleOutput(result="included", score=5).model_dump_json(),
            "other_key": SimpleOutput(result="excluded", score=9).model_dump_json(),
        }
        ctx = MockInvocationContext(state)
        
        agg = SimpleAggregator(
            name="test",
            input_keys=["key1"],  # Only key1
            input_schema=SimpleOutput,
            output_key="result",
        )
        
        results = agg._collect_results(ctx)
        
        assert len(results) == 1
        assert results[0].result == "included"


class TestAggregatorAgentAggregationFn:
    """Test AggregatorAgent.aggregation_fn behavior."""
    
    @pytest.mark.asyncio
    async def test_aggregation_fn_receives_parsed_results(self):
        """Test that aggregation_fn receives parsed Pydantic models."""
        received_results = []
        
        class TrackingAggregator(AggregatorAgent[SimpleOutput]):
            async def aggregation_fn(self, results):
                received_results.extend(results)
                return results[0] if results else None
        
        agg = TrackingAggregator(
            name="test",
            input_keys=[],
            input_schema=SimpleOutput,
            output_key="result",
        )
        
        test_results = [
            SimpleOutput(result="a", score=1),
            SimpleOutput(result="b", score=2),
        ]
        
        await agg.aggregation_fn(test_results)
        
        assert len(received_results) == 2
        assert all(isinstance(r, SimpleOutput) for r in received_results)
    
    @pytest.mark.asyncio
    async def test_aggregation_fn_can_return_single_result(self):
        """Test aggregation_fn returning a single result."""
        class MaxScoreAggregator(AggregatorAgent[SimpleOutput]):
            async def aggregation_fn(self, results):
                return max(results, key=lambda x: x.score) if results else None
        
        agg = MaxScoreAggregator(
            name="test",
            input_keys=[],
            input_schema=SimpleOutput,
            output_key="result",
        )
        
        results = [
            SimpleOutput(result="low", score=3),
            SimpleOutput(result="high", score=9),
            SimpleOutput(result="mid", score=5),
        ]
        
        best = await agg.aggregation_fn(results)
        
        assert best.result == "high"
        assert best.score == 9
    
    @pytest.mark.asyncio
    async def test_aggregation_fn_can_return_none(self):
        """Test aggregation_fn returning None."""
        class NoneAggregator(AggregatorAgent[SimpleOutput]):
            async def aggregation_fn(self, results):
                return None
        
        agg = NoneAggregator(
            name="test",
            input_keys=[],
            input_schema=SimpleOutput,
            output_key="result",
        )
        
        result = await agg.aggregation_fn([SimpleOutput(result="test", score=5)])
        
        assert result is None


# ============================================================================
# MapReduceAgent Tests
# ============================================================================


class TestMapReduceAgentConstruction:
    """Test MapReduceAgent construction."""
    
    def test_map_reduce_agent_creation(self):
        """Test MapReduceAgent can be created."""
        class SimpleMapper(MapperAgent):
            def build_items_from_state(self, ctx):
                return []
        
        mapper = SimpleMapper(worker_factory=lambda x: MockStatefulWorker(x))
        
        agent = MapReduceAgent(
            mapper=mapper,
            aggregator_factory=lambda keys: None,
            name="test_mr",
        )
        
        assert agent.name == "test_mr"
        assert agent._mapper is mapper
    
    def test_map_reduce_agent_default_name(self):
        """Test MapReduceAgent default name."""
        class SimpleMapper(MapperAgent):
            def build_items_from_state(self, ctx):
                return []
        
        mapper = SimpleMapper(worker_factory=lambda x: MockStatefulWorker(x))
        
        agent = MapReduceAgent(
            mapper=mapper,
            aggregator_factory=lambda keys: None,
        )
        
        assert agent.name == "map_reduce"
    
    def test_map_reduce_stores_aggregator_factory(self):
        """Test that MapReduceAgent stores the aggregator factory."""
        class SimpleMapper(MapperAgent):
            def build_items_from_state(self, ctx):
                return []
        
        mapper = SimpleMapper(worker_factory=lambda x: MockStatefulWorker(x))
        factory = lambda keys: None
        
        agent = MapReduceAgent(
            mapper=mapper,
            aggregator_factory=factory,
        )
        
        assert agent._aggregator_factory is factory


# ============================================================================
# SentenceVideoMapper Tests
# ============================================================================


class TestSentenceVideoMapper:
    """Test SentenceVideoMapper concrete implementation."""
    
    def test_mapper_creation(self):
        """Test SentenceVideoMapper can be created."""
        retriever = MockVideoRetriever()
        mapper = SentenceVideoMapper(
            video_retriever=retriever,
            max_candidates=5,
        )
        
        assert mapper._video_retriever is retriever
        assert mapper._max_candidates == 5
    
    def test_mapper_default_values(self):
        """Test SentenceVideoMapper default values."""
        retriever = MockVideoRetriever()
        mapper = SentenceVideoMapper(video_retriever=retriever)
        
        assert mapper._max_candidates == 5  # Default
        assert mapper.name == "sentence_video_mapper"  # Default
    
    def test_mapper_custom_name(self):
        """Test SentenceVideoMapper with custom name."""
        retriever = MockVideoRetriever()
        mapper = SentenceVideoMapper(
            video_retriever=retriever,
            name="custom_mapper",
        )
        
        assert mapper.name == "custom_mapper"
    
    def test_build_items_from_state_with_sentences(self):
        """Test build_items_from_state generates correct items."""
        video_map = {
            "Hello": ["/v1.mp4", "/v2.mp4"],
            "World": ["/v3.mp4"],
        }
        retriever = MockVideoRetriever(video_map)
        mapper = SentenceVideoMapper(
            video_retriever=retriever,
            max_candidates=10,
        )
        
        ctx = MockInvocationContext({
            SENTENCES_KEY: ["Hello", "World"],
        })
        
        items = mapper.build_items_from_state(ctx)
        
        # Should have 3 items total (2 for Hello, 1 for World)
        assert len(items) == 3
        
        # Check structure
        hello_items = [i for i in items if i["sentence"] == "Hello"]
        world_items = [i for i in items if i["sentence"] == "World"]
        
        assert len(hello_items) == 2
        assert len(world_items) == 1
        
        assert hello_items[0]["video_path"] == "/v1.mp4"
        assert hello_items[1]["video_path"] == "/v2.mp4"
    
    def test_build_items_respects_max_candidates(self):
        """Test that max_candidates limits videos per sentence."""
        retriever = MockVideoRetriever({
            "test": ["/v1.mp4", "/v2.mp4", "/v3.mp4", "/v4.mp4", "/v5.mp4"]
        })
        mapper = SentenceVideoMapper(
            video_retriever=retriever,
            max_candidates=2,
        )
        
        ctx = MockInvocationContext({SENTENCES_KEY: ["test"]})
        items = mapper.build_items_from_state(ctx)
        
        assert len(items) == 2
    
    def test_build_items_empty_sentences(self):
        """Test build_items with no sentences."""
        retriever = MockVideoRetriever()
        mapper = SentenceVideoMapper(video_retriever=retriever)
        
        ctx = MockInvocationContext({SENTENCES_KEY: []})
        items = mapper.build_items_from_state(ctx)
        
        assert items == []
    
    def test_build_items_missing_sentences_key(self):
        """Test build_items when sentences key is missing."""
        retriever = MockVideoRetriever()
        mapper = SentenceVideoMapper(video_retriever=retriever)
        
        ctx = MockInvocationContext({})
        items = mapper.build_items_from_state(ctx)
        
        assert items == []
    
    def test_build_items_handles_no_candidates(self):
        """Test build_items when retriever returns no candidates."""
        retriever = MockVideoRetriever({"no_results": []})
        mapper = SentenceVideoMapper(video_retriever=retriever)
        
        ctx = MockInvocationContext({SENTENCES_KEY: ["no_results"]})
        items = mapper.build_items_from_state(ctx)
        
        # Should skip sentences with no candidates
        assert items == []
    
    def test_build_items_calls_retriever(self):
        """Test that build_items calls the video retriever."""
        retriever = MockVideoRetriever()
        mapper = SentenceVideoMapper(
            video_retriever=retriever,
            max_candidates=3,
        )
        
        ctx = MockInvocationContext({SENTENCES_KEY: ["query1", "query2"]})
        mapper.build_items_from_state(ctx)
        
        assert len(retriever.search_calls) == 2
        assert retriever.search_calls[0] == ("query1", 3)
        assert retriever.search_calls[1] == ("query2", 3)


# ============================================================================
# SentenceVideoAggregator Tests
# ============================================================================


class TestSentenceVideoAggregator:
    """Test SentenceVideoAggregator concrete implementation."""
    
    def test_aggregator_creation(self):
        """Test SentenceVideoAggregator can be created."""
        agg = SentenceVideoAggregator(
            input_keys=["key1", "key2"],
            output_key="matches",
        )
        
        assert agg._input_keys == ["key1", "key2"]
        assert agg._output_key == "matches"
    
    def test_aggregator_default_output_key(self):
        """Test SentenceVideoAggregator default output key."""
        agg = SentenceVideoAggregator(input_keys=[])
        
        assert agg._output_key == MATCHES_KEY
    
    @pytest.mark.asyncio
    async def test_aggregation_groups_by_sentence(self):
        """Test that aggregation groups results by sentence."""
        results = [
            VideoMatchResult(
                sentence="Hello",
                video_path="/v1.mp4",
                rating=ContextualRating.CONTEXTUAL,
                grade=8,
                reasoning="Good"
            ),
            VideoMatchResult(
                sentence="Hello",
                video_path="/v2.mp4",
                rating=ContextualRating.NEUTRAL,
                grade=5,
                reasoning="OK"
            ),
            VideoMatchResult(
                sentence="World",
                video_path="/v3.mp4",
                rating=ContextualRating.CONTEXTUAL,
                grade=7,
                reasoning="Good"
            ),
        ]
        
        agg = SentenceVideoAggregator(input_keys=[], output_key="out")
        output = await agg.aggregation_fn(results)
        
        assert isinstance(output, SentenceVideoMatcherOutput)
        assert len(output.matches) == 2  # One per sentence
        
        # Check correct best was selected per sentence
        hello_match = next(m for m in output.matches if m.sentence == "Hello")
        world_match = next(m for m in output.matches if m.sentence == "World")
        
        assert hello_match.video_path == "/v1.mp4"  # CONTEXTUAL wins
        assert world_match.video_path == "/v3.mp4"
    
    @pytest.mark.asyncio
    async def test_aggregation_selects_best_by_rating_priority(self):
        """Test that CONTEXTUAL > NEUTRAL > NOT_CONTEXTUAL."""
        results = [
            VideoMatchResult(
                sentence="test",
                video_path="/v1.mp4",
                rating=ContextualRating.NOT_CONTEXTUAL,
                grade=10,  # Highest grade
                reasoning="Bad"
            ),
            VideoMatchResult(
                sentence="test",
                video_path="/v2.mp4",
                rating=ContextualRating.CONTEXTUAL,
                grade=3,  # Lower grade but better rating
                reasoning="Good"
            ),
        ]
        
        agg = SentenceVideoAggregator(input_keys=[], output_key="out")
        output = await agg.aggregation_fn(results)
        
        # Should select CONTEXTUAL despite lower grade
        assert output.matches[0].video_path == "/v2.mp4"
        assert output.matches[0].rating == ContextualRating.CONTEXTUAL
    
    @pytest.mark.asyncio
    async def test_aggregation_neutral_over_not_contextual(self):
        """Test that NEUTRAL beats NOT_CONTEXTUAL."""
        results = [
            VideoMatchResult(
                sentence="test",
                video_path="/v1.mp4",
                rating=ContextualRating.NOT_CONTEXTUAL,
                grade=10,
                reasoning="High grade but bad rating"
            ),
            VideoMatchResult(
                sentence="test",
                video_path="/v2.mp4",
                rating=ContextualRating.NEUTRAL,
                grade=4,
                reasoning="Lower grade but better rating"
            ),
        ]
        
        agg = SentenceVideoAggregator(input_keys=[], output_key="out")
        output = await agg.aggregation_fn(results)
        
        assert output.matches[0].rating == ContextualRating.NEUTRAL
        assert output.matches[0].video_path == "/v2.mp4"
    
    @pytest.mark.asyncio
    async def test_aggregation_selects_highest_grade_within_rating(self):
        """Test that highest grade is selected within same rating."""
        results = [
            VideoMatchResult(
                sentence="test",
                video_path="/v1.mp4",
                rating=ContextualRating.CONTEXTUAL,
                grade=5,
                reasoning="Lower"
            ),
            VideoMatchResult(
                sentence="test",
                video_path="/v2.mp4",
                rating=ContextualRating.CONTEXTUAL,
                grade=9,
                reasoning="Higher"
            ),
        ]
        
        agg = SentenceVideoAggregator(input_keys=[], output_key="out")
        output = await agg.aggregation_fn(results)
        
        assert output.matches[0].grade == 9
        assert output.matches[0].video_path == "/v2.mp4"
    
    @pytest.mark.asyncio
    async def test_aggregation_fallback_to_highest_grade(self):
        """Test fallback to highest grade when only NOT_CONTEXTUAL."""
        results = [
            VideoMatchResult(
                sentence="test",
                video_path="/v1.mp4",
                rating=ContextualRating.NOT_CONTEXTUAL,
                grade=3,
                reasoning="Low"
            ),
            VideoMatchResult(
                sentence="test",
                video_path="/v2.mp4",
                rating=ContextualRating.NOT_CONTEXTUAL,
                grade=8,
                reasoning="High"
            ),
        ]
        
        agg = SentenceVideoAggregator(input_keys=[], output_key="out")
        output = await agg.aggregation_fn(results)
        
        assert output.matches[0].grade == 8
        assert output.matches[0].video_path == "/v2.mp4"
    
    @pytest.mark.asyncio
    async def test_aggregation_empty_results(self):
        """Test aggregation with empty results."""
        agg = SentenceVideoAggregator(input_keys=[], output_key="out")
        output = await agg.aggregation_fn([])
        
        assert isinstance(output, SentenceVideoMatcherOutput)
        assert output.matches == []
    
    @pytest.mark.asyncio
    async def test_aggregation_multiple_sentences(self):
        """Test aggregation with many sentences."""
        results = []
        for i in range(5):
            for j in range(3):  # 3 videos per sentence
                results.append(VideoMatchResult(
                    sentence=f"Sentence {i}",
                    video_path=f"/video_{i}_{j}.mp4",
                    rating=ContextualRating.CONTEXTUAL if j == 0 else ContextualRating.NEUTRAL,
                    grade=10 - j,
                    reasoning=f"Result {j}"
                ))
        
        agg = SentenceVideoAggregator(input_keys=[], output_key="out")
        output = await agg.aggregation_fn(results)
        
        assert len(output.matches) == 5  # One per sentence
        
        # Each should have the first video (j=0, CONTEXTUAL, grade=10)
        for i, match in enumerate(output.matches):
            assert match.rating == ContextualRating.CONTEXTUAL
            assert match.grade == 10


# ============================================================================
# SentenceVideoMatcherAgent Integration Tests
# ============================================================================


class TestSentenceVideoMatcherAgentIntegration:
    """Integration tests for SentenceVideoMatcherAgent."""
    
    def test_agent_creation_via_class(self):
        """Test creating agent via class constructor."""
        retriever = MockVideoRetriever()
        agent = SentenceVideoMatcherAgent(
            video_retriever=retriever,
            max_candidates=5,
        )
        
        assert agent._video_retriever is retriever
        assert agent._max_candidates == 5
        assert agent.name == "sentence_video_matcher"
    
    def test_agent_creation_via_factory(self):
        """Test creating agent via factory function."""
        retriever = MockVideoRetriever()
        agent = create_sentence_video_matcher(
            video_retriever=retriever,
            max_candidates=3,
            name="custom_name",
        )
        
        assert agent.name == "custom_name"
        assert isinstance(agent, MapReduceAgent)
    
    def test_agent_is_map_reduce_agent(self):
        """Test that SentenceVideoMatcherAgent is a MapReduceAgent."""
        retriever = MockVideoRetriever()
        agent = SentenceVideoMatcherAgent(video_retriever=retriever)
        
        assert isinstance(agent, MapReduceAgent)
    
    def test_agent_custom_name(self):
        """Test agent with custom name."""
        retriever = MockVideoRetriever()
        agent = SentenceVideoMatcherAgent(
            video_retriever=retriever,
            name="my_matcher",
        )
        
        assert agent.name == "my_matcher"
    
    def test_agent_has_mapper(self):
        """Test that agent has internal mapper."""
        retriever = MockVideoRetriever()
        agent = SentenceVideoMatcherAgent(video_retriever=retriever)
        
        assert hasattr(agent, "_mapper")
        assert isinstance(agent._mapper, SentenceVideoMapper)
    
    def test_agent_mapper_has_retriever(self):
        """Test that agent's mapper has the video retriever."""
        retriever = MockVideoRetriever()
        agent = SentenceVideoMatcherAgent(
            video_retriever=retriever,
            max_candidates=7,
        )
        
        assert agent._mapper._video_retriever is retriever
        assert agent._mapper._max_candidates == 7


# ============================================================================
# Backward Compatibility Tests
# ============================================================================


class TestBackwardCompatibility:
    """Test backward compatibility with old interfaces."""
    
    def test_abstract_aggregator_alias(self):
        """Test that AbstractAggregator is aliased to AggregatorAgent."""
        from virtual_streamer.lib.agents import AbstractAggregator
        
        assert AbstractAggregator is AggregatorAgent
    
    def test_best_match_aggregator_uses_new_interface(self):
        """Test that BestMatchAggregator works with new interface."""
        from virtual_streamer.agents.video_matcher.aggregator import BestMatchAggregator
        
        agg = BestMatchAggregator(
            input_keys=["key1", "key2"],
            output_key="best",
        )
        
        assert agg._input_keys == ["key1", "key2"]
        assert agg._output_key == "best"
    
    def test_aggregator_agent_exported(self):
        """Test that AggregatorAgent is exported from lib.agents."""
        from virtual_streamer.lib.agents import AggregatorAgent as Exported
        
        assert Exported is AggregatorAgent
    
    def test_mapper_agent_exported(self):
        """Test that MapperAgent is exported from lib.agents."""
        from virtual_streamer.lib.agents import MapperAgent as Exported
        
        assert Exported is MapperAgent
    
    def test_map_reduce_agent_exported(self):
        """Test that MapReduceAgent is exported from lib.agents."""
        from virtual_streamer.lib.agents import MapReduceAgent as Exported
        
        assert Exported is MapReduceAgent


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
