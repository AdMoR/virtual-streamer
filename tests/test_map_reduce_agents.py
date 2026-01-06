"""
Tests for MapReduceAgent architecture.

Tests cover:
- MapperAgent abstract base class
- AggregatorAgent abstract base class
- MapReduceAgent orchestration
- BestMatchAggregator concrete implementation
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
# BestMatchAggregator Tests
# ============================================================================


class TestBestMatchAggregator:
    """Test BestMatchAggregator concrete implementation."""
    
    def test_aggregator_creation(self):
        """Test BestMatchAggregator can be created."""
        from virtual_streamer.agents.video_matcher.aggregator import BestMatchAggregator
        
        agg = BestMatchAggregator(
            input_keys=["key1", "key2"],
            output_key="best",
        )
        
        assert agg._input_keys == ["key1", "key2"]
        assert agg._output_key == "best"
    
    def test_aggregator_default_output_key(self):
        """Test BestMatchAggregator default output key."""
        from virtual_streamer.agents.video_matcher.aggregator import BestMatchAggregator
        
        agg = BestMatchAggregator(input_keys=[])
        
        assert agg._output_key == "best_video_match"
    
    @pytest.mark.asyncio
    async def test_aggregation_selects_contextual_over_neutral(self):
        """Test that CONTEXTUAL > NEUTRAL > NOT_CONTEXTUAL."""
        from virtual_streamer.agents.video_matcher.aggregator import BestMatchAggregator
        from virtual_streamer.agents.video_matcher.schema import (
            VideoMatchResult,
            ContextualRating,
        )
        
        results = [
            VideoMatchResult(
                line_id=0,
                character_id="narrator",
                sentence="test",
                scene_description="A test scene",
                video_path="/v1.mp4",
                rating=ContextualRating.NOT_CONTEXTUAL,
                grade=10,  # Highest grade
                reasoning="Bad"
            ),
            VideoMatchResult(
                line_id=0,
                character_id="narrator",
                sentence="test",
                scene_description="A test scene",
                video_path="/v2.mp4",
                rating=ContextualRating.CONTEXTUAL,
                grade=3,  # Lower grade but better rating
                reasoning="Good"
            ),
        ]
        
        agg = BestMatchAggregator(input_keys=[], output_key="out")
        best = await agg.aggregation_fn(results)
        
        # Should select CONTEXTUAL despite lower grade
        assert best.video_path == "/v2.mp4"
        assert best.rating == ContextualRating.CONTEXTUAL
    
    @pytest.mark.asyncio
    async def test_aggregation_selects_highest_grade_within_rating(self):
        """Test that highest grade is selected within same rating."""
        from virtual_streamer.agents.video_matcher.aggregator import BestMatchAggregator
        from virtual_streamer.agents.video_matcher.schema import (
            VideoMatchResult,
            ContextualRating,
        )
        
        results = [
            VideoMatchResult(
                line_id=0,
                character_id="narrator",
                sentence="test",
                scene_description="A test scene",
                video_path="/v1.mp4",
                rating=ContextualRating.CONTEXTUAL,
                grade=5,
                reasoning="Lower"
            ),
            VideoMatchResult(
                line_id=0,
                character_id="narrator",
                sentence="test",
                scene_description="A test scene",
                video_path="/v2.mp4",
                rating=ContextualRating.CONTEXTUAL,
                grade=9,
                reasoning="Higher"
            ),
        ]
        
        agg = BestMatchAggregator(input_keys=[], output_key="out")
        best = await agg.aggregation_fn(results)
        
        assert best.grade == 9
        assert best.video_path == "/v2.mp4"
    
    @pytest.mark.asyncio
    async def test_aggregation_empty_results(self):
        """Test aggregation with empty results."""
        from virtual_streamer.agents.video_matcher.aggregator import BestMatchAggregator
        
        agg = BestMatchAggregator(input_keys=[], output_key="out")
        result = await agg.aggregation_fn([])
        
        assert result is None


# ============================================================================
# Backward Compatibility Tests
# ============================================================================


class TestBackwardCompatibility:
    """Test backward compatibility with old interfaces."""
    
    def test_abstract_aggregator_alias(self):
        """Test that AbstractAggregator is aliased to AggregatorAgent."""
        from virtual_streamer.lib.agents import AbstractAggregator
        
        assert AbstractAggregator is AggregatorAgent
    
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
