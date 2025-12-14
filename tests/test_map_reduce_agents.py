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
from unittest.mock import MagicMock, AsyncMock

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
    
    async def run_async(self, ctx):
        """Simulate worker execution by writing to state."""
        # Read input
        input_data = ctx.session.state.get(self.get_input_key())
        if input_data:
            parsed = self._input_schema.model_validate_json(input_data)
            # Write output
            output = self._output_schema(result=f"processed_{parsed.value}", score=5)
            ctx.session.state[self.get_output_key()] = output.model_dump_json()
        yield  # Make it an async generator


class MockInvocationContext:
    """Mock invocation context with session state."""
    
    def __init__(self, initial_state: Optional[Dict[str, Any]] = None):
        self.session = MagicMock()
        self.session.state = initial_state or {}


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


class TestMapperAgentExecution:
    """Test MapperAgent execution behavior."""
    
    @pytest.mark.asyncio
    async def test_mapper_builds_items_from_state(self):
        """Test that mapper calls build_items_from_state."""
        build_calls = []
        
        class TrackingMapper(MapperAgent):
            def build_items_from_state(self, ctx):
                build_calls.append(ctx)
                return [{"value": "item1"}, {"value": "item2"}]
        
        ctx = MockInvocationContext({"input_data": "test"})
        mapper = TrackingMapper(worker_factory=lambda x: MockStatefulWorker(x))
        
        events = []
        async for event in mapper.run_async(ctx):
            events.append(event)
        
        assert len(build_calls) == 1
        assert build_calls[0] is ctx
    
    @pytest.mark.asyncio
    async def test_mapper_creates_workers_for_each_item(self):
        """Test that mapper creates one worker per item."""
        created_workers = []
        
        def track_factory(run_id):
            worker = MockStatefulWorker(run_id)
            created_workers.append(worker)
            return worker
        
        class ItemMapper(MapperAgent):
            def build_items_from_state(self, ctx):
                return [{"value": "a"}, {"value": "b"}, {"value": "c"}]
        
        ctx = MockInvocationContext()
        mapper = ItemMapper(worker_factory=track_factory)
        
        async for _ in mapper.run_async(ctx):
            pass
        
        assert len(created_workers) == 3
    
    @pytest.mark.asyncio
    async def test_mapper_writes_items_to_state(self):
        """Test that mapper writes validated items to state."""
        class ItemMapper(MapperAgent):
            def build_items_from_state(self, ctx):
                return [{"value": "test_value"}]
        
        ctx = MockInvocationContext()
        mapper = ItemMapper(worker_factory=lambda x: MockStatefulWorker(x))
        
        async for event in mapper.run_async(ctx):
            if event.actions and event.actions.state_delta:
                # Check that item was written to state
                for key, value in event.actions.state_delta.items():
                    if key.startswith("task:"):
                        assert "test_value" in value
    
    @pytest.mark.asyncio
    async def test_mapper_get_output_keys_after_run(self):
        """Test that get_output_keys works after run."""
        class ItemMapper(MapperAgent):
            def build_items_from_state(self, ctx):
                return [{"value": "a"}, {"value": "b"}]
        
        ctx = MockInvocationContext()
        mapper = ItemMapper(worker_factory=lambda x: MockStatefulWorker(x))
        
        async for _ in mapper.run_async(ctx):
            pass
        
        output_keys = mapper.get_output_keys()
        assert len(output_keys) == 2
        assert all("result:" in k for k in output_keys)
    
    @pytest.mark.asyncio
    async def test_mapper_handles_empty_items(self):
        """Test that mapper handles empty items gracefully."""
        class EmptyMapper(MapperAgent):
            def build_items_from_state(self, ctx):
                return []
        
        ctx = MockInvocationContext()
        mapper = EmptyMapper(worker_factory=lambda x: MockStatefulWorker(x))
        
        events = []
        async for event in mapper.run_async(ctx):
            events.append(event)
        
        # Should emit at least one event about no items
        assert len(events) >= 1
        assert mapper.get_output_keys() == []


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


class TestAggregatorAgentExecution:
    """Test AggregatorAgent execution behavior."""
    
    @pytest.mark.asyncio
    async def test_aggregator_collects_from_input_keys(self):
        """Test that aggregator reads from specified input keys."""
        collected = []
        
        class TrackingAggregator(AggregatorAgent[SimpleOutput]):
            async def aggregation_fn(self, results):
                collected.extend(results)
                return results[0] if results else None
        
        state = {
            "result:w0": SimpleOutput(result="r1", score=5).model_dump_json(),
            "result:w1": SimpleOutput(result="r2", score=8).model_dump_json(),
            "other_key": SimpleOutput(result="ignored", score=1).model_dump_json(),
        }
        ctx = MockInvocationContext(state)
        
        agg = TrackingAggregator(
            name="test",
            input_keys=["result:w0", "result:w1"],
            input_schema=SimpleOutput,
            output_key="final",
        )
        
        async for _ in agg.run_async(ctx):
            pass
        
        assert len(collected) == 2
        assert collected[0].result == "r1"
        assert collected[1].result == "r2"
    
    @pytest.mark.asyncio
    async def test_aggregator_writes_to_output_key(self):
        """Test that aggregator writes result to output key."""
        class SumAggregator(AggregatorAgent[SimpleOutput]):
            async def aggregation_fn(self, results):
                total = sum(r.score for r in results)
                return SimpleOutput(result="sum", score=min(total, 10))
        
        state = {
            "key1": SimpleOutput(result="a", score=3).model_dump_json(),
            "key2": SimpleOutput(result="b", score=4).model_dump_json(),
        }
        ctx = MockInvocationContext(state)
        
        agg = SumAggregator(
            name="sum",
            input_keys=["key1", "key2"],
            input_schema=SimpleOutput,
            output_key="total",
        )
        
        async for event in agg.run_async(ctx):
            if event.actions and event.actions.state_delta:
                assert "total" in event.actions.state_delta
    
    @pytest.mark.asyncio
    async def test_aggregator_handles_missing_keys(self):
        """Test that aggregator handles missing keys gracefully."""
        collected = []
        
        class TrackingAggregator(AggregatorAgent[SimpleOutput]):
            async def aggregation_fn(self, results):
                collected.extend(results)
                return results[0] if results else None
        
        state = {
            "key1": SimpleOutput(result="exists", score=5).model_dump_json(),
            # key2 is missing
        }
        ctx = MockInvocationContext(state)
        
        agg = TrackingAggregator(
            name="test",
            input_keys=["key1", "key2", "key3"],
            input_schema=SimpleOutput,
            output_key="result",
        )
        
        async for _ in agg.run_async(ctx):
            pass
        
        # Should only collect the one existing key
        assert len(collected) == 1
    
    @pytest.mark.asyncio
    async def test_aggregator_handles_empty_results(self):
        """Test that aggregator emits escalate when no results."""
        class SimpleAggregator(AggregatorAgent[SimpleOutput]):
            async def aggregation_fn(self, results):
                return results[0] if results else None
        
        ctx = MockInvocationContext({})
        
        agg = SimpleAggregator(
            name="test",
            input_keys=["missing1", "missing2"],
            input_schema=SimpleOutput,
            output_key="result",
        )
        
        events = []
        async for event in agg.run_async(ctx):
            events.append(event)
        
        # Should escalate when no results
        assert any(e.actions and e.actions.escalate for e in events)


# ============================================================================
# MapReduceAgent Tests
# ============================================================================


class TestMapReduceAgentOrchestration:
    """Test MapReduceAgent orchestration."""
    
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
    
    @pytest.mark.asyncio
    async def test_map_reduce_runs_mapper_first(self):
        """Test that MapReduceAgent runs mapper before aggregator."""
        execution_order = []
        
        class OrderMapper(MapperAgent):
            def build_items_from_state(self, ctx):
                execution_order.append("mapper")
                return [{"value": "test"}]
        
        class OrderAggregator(AggregatorAgent[SimpleOutput]):
            async def aggregation_fn(self, results):
                execution_order.append("aggregator")
                return None
        
        def agg_factory(keys):
            execution_order.append("factory")
            return OrderAggregator(
                name="agg",
                input_keys=keys,
                input_schema=SimpleOutput,
                output_key="result",
            )
        
        ctx = MockInvocationContext()
        mapper = OrderMapper(worker_factory=lambda x: MockStatefulWorker(x))
        
        agent = MapReduceAgent(
            mapper=mapper,
            aggregator_factory=agg_factory,
        )
        
        async for _ in agent.run_async(ctx):
            pass
        
        # Mapper should run, then factory called, then aggregator
        assert execution_order.index("mapper") < execution_order.index("factory")
    
    @pytest.mark.asyncio
    async def test_map_reduce_wires_output_keys(self):
        """Test that MapReduceAgent passes mapper output keys to aggregator."""
        received_keys = []
        
        class SimpleMapper(MapperAgent):
            def build_items_from_state(self, ctx):
                return [{"value": "a"}, {"value": "b"}]
        
        class KeyTrackingAggregator(AggregatorAgent[SimpleOutput]):
            async def aggregation_fn(self, results):
                return None
        
        def tracking_factory(keys):
            received_keys.extend(keys)
            return KeyTrackingAggregator(
                name="agg",
                input_keys=keys,
                input_schema=SimpleOutput,
                output_key="result",
            )
        
        ctx = MockInvocationContext()
        mapper = SimpleMapper(worker_factory=lambda x: MockStatefulWorker(x))
        
        agent = MapReduceAgent(
            mapper=mapper,
            aggregator_factory=tracking_factory,
        )
        
        async for _ in agent.run_async(ctx):
            pass
        
        # Should have received 2 keys (one per item)
        assert len(received_keys) == 2
        assert all("result:" in k for k in received_keys)
    
    @pytest.mark.asyncio
    async def test_map_reduce_handles_empty_mapper(self):
        """Test MapReduceAgent handles mapper with no items."""
        aggregator_created = []
        
        class EmptyMapper(MapperAgent):
            def build_items_from_state(self, ctx):
                return []
        
        def tracking_factory(keys):
            aggregator_created.append(keys)
            return None
        
        ctx = MockInvocationContext()
        mapper = EmptyMapper(worker_factory=lambda x: MockStatefulWorker(x))
        
        agent = MapReduceAgent(
            mapper=mapper,
            aggregator_factory=tracking_factory,
        )
        
        async for _ in agent.run_async(ctx):
            pass
        
        # Aggregator factory should NOT be called if no output keys
        assert len(aggregator_created) == 0


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
    async def test_aggregation_empty_results(self):
        """Test aggregation with empty results."""
        agg = SentenceVideoAggregator(input_keys=[], output_key="out")
        output = await agg.aggregation_fn([])
        
        assert isinstance(output, SentenceVideoMatcherOutput)
        assert output.matches == []


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

