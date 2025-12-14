"""
Verification tests for attribute assignment patterns with ADK agents.

This test file verifies which attribute assignment patterns work with
Google ADK's Pydantic-based agents (BaseAgent, ParallelAgent, LlmAgent).

## FINDINGS (after running tests):

### ✅ WORKING PATTERNS:
1. object.__setattr__(self, 'attr', value)     - Current workaround
2. self._attr = value                          - Underscore prefix (cleanest!)
3. Pydantic field declarations with ConfigDict - Most type-safe

### ❌ BROKEN PATTERNS:
1. self.attr = value                           - Direct assignment FAILS
   -> Raises: ValueError: "Agent" object has no field "attr"

### WHY UNDERSCORE WORKS:
Pydantic's _fields.is_valid_field_name() returns False for names starting
with underscore, so they bypass Pydantic's field validation entirely.

### RECOMMENDATION:
Use underscore prefix `self._attr = value` for custom attributes.
It's Pythonic, clean, and happens to bypass Pydantic's validation.
"""

import pytest
from typing import List, Any


class TestBaseAgentAttributeAssignment:
    """Verify attribute assignment patterns with ADK BaseAgent."""
    
    def test_direct_assignment_after_init_fails(self):
        """Test that direct assignment FAILS (expected behavior)."""
        from google.adk.agents import BaseAgent
        
        class TestAgent(BaseAgent):
            def __init__(self, custom_attr: str):
                super().__init__(name="test_agent")
                self.custom_attr = custom_attr  # This will fail!
        
        with pytest.raises(ValueError, match='has no field "custom_attr"'):
            TestAgent(custom_attr="test_value")
    
    def test_object_setattr_after_init(self):
        """Test object.__setattr__ pattern (current workaround) - WORKS."""
        from google.adk.agents import BaseAgent
        
        class TestAgent(BaseAgent):
            def __init__(self, custom_attr: str):
                super().__init__(name="test_agent")
                object.__setattr__(self, 'custom_attr', custom_attr)
        
        agent = TestAgent(custom_attr="test_value")
        
        assert agent.custom_attr == "test_value"
        assert agent.name == "test_agent"
    
    def test_underscore_prefix_assignment(self):
        """Test underscore-prefixed attributes - WORKS (recommended!)."""
        from google.adk.agents import BaseAgent
        
        class CustomService:
            def __init__(self, name: str):
                self.name = name
            
            def do_something(self) -> str:
                return f"Done by {self.name}"
        
        class TestAgent(BaseAgent):
            def __init__(self, service: CustomService, config: dict):
                super().__init__(name="test_agent")
                self._service = service  # Underscore prefix bypasses Pydantic!
                self._config = config
            
            @property
            def service(self) -> CustomService:
                """Public accessor for private attribute."""
                return self._service
        
        service = CustomService("my_service")
        agent = TestAgent(service=service, config={"key": "value"})
        
        assert agent._service is service
        assert agent.service.do_something() == "Done by my_service"
        assert agent._config == {"key": "value"}
    
    def test_complex_types_direct_assignment_fails(self):
        """Test that direct assignment with complex types FAILS."""
        from google.adk.agents import BaseAgent
        
        class CustomService:
            pass
        
        class TestAgent(BaseAgent):
            def __init__(self, service: CustomService):
                super().__init__(name="test_agent")
                self.service = service  # This will fail!
        
        with pytest.raises(ValueError, match='has no field "service"'):
            TestAgent(service=CustomService())


class TestParallelAgentAttributeAssignment:
    """Verify attribute assignment patterns with ADK ParallelAgent."""
    
    def test_direct_assignment_after_init_fails(self):
        """Test that direct assignment FAILS with ParallelAgent."""
        from google.adk.agents import ParallelAgent
        
        class TestParallel(ParallelAgent):
            def __init__(self, items: List[Any]):
                super().__init__(name="test_parallel", sub_agents=[])
                self.items = items  # This will fail!
        
        with pytest.raises(ValueError, match='has no field "items"'):
            TestParallel(items=[1, 2, 3])
    
    def test_object_setattr_after_init(self):
        """Test object.__setattr__ pattern with ParallelAgent - WORKS."""
        from google.adk.agents import ParallelAgent
        
        class TestParallel(ParallelAgent):
            def __init__(self, items: List[Any]):
                super().__init__(name="test_parallel", sub_agents=[])
                object.__setattr__(self, 'items', items)
        
        agent = TestParallel(items=[1, 2, 3])
        
        assert agent.items == [1, 2, 3]
    
    def test_underscore_prefix_assignment(self):
        """Test underscore-prefixed attributes with ParallelAgent - WORKS."""
        from google.adk.agents import ParallelAgent
        
        class TestParallel(ParallelAgent):
            def __init__(self, items: List[Any], worker_factory):
                super().__init__(name="test_parallel", sub_agents=[])
                self._items = items
                self._worker_factory = worker_factory
        
        factory = lambda x: x
        agent = TestParallel(items=[1, 2, 3], worker_factory=factory)
        
        assert agent._items == [1, 2, 3]
        assert agent._worker_factory is factory
        assert agent.name == "test_parallel"


class TestLlmAgentAttributeAssignment:
    """Verify attribute assignment patterns with LlmAgent (via BaseLlmAgent)."""
    
    def test_direct_assignment_stateful_pattern(self):
        """Test direct assignment mimicking StatefulLlmAgent pattern."""
        from google.adk.agents import BaseAgent
        
        class MockCallback:
            def __init__(self, run_id: str):
                self.run_id = run_id
            
            def get_key(self) -> str:
                return f"key:{self.run_id}"
        
        class TestStatefulAgent(BaseAgent):
            def __init__(self, input_callback: MockCallback, output_callback: MockCallback):
                super().__init__(name="stateful_agent")
                self._input_callback = input_callback
                self._output_callback = output_callback
            
            def get_input_key(self) -> str:
                return self._input_callback.get_key()
            
            def get_output_key(self) -> str:
                return self._output_callback.get_key()
        
        input_cb = MockCallback("run1")
        output_cb = MockCallback("run1")
        agent = TestStatefulAgent(input_callback=input_cb, output_callback=output_cb)
        
        assert agent._input_callback is input_cb
        assert agent._output_callback is output_cb
        assert agent.get_input_key() == "key:run1"
        assert agent.get_output_key() == "key:run1"


class TestPydanticFieldDeclaration:
    """Test if Pydantic field declarations work as an alternative."""
    
    def test_with_config_dict(self):
        """Test using ConfigDict to allow arbitrary types."""
        from google.adk.agents import BaseAgent
        from pydantic import ConfigDict
        
        class CustomService:
            pass
        
        class TestAgent(BaseAgent):
            model_config = ConfigDict(arbitrary_types_allowed=True)
            
            custom_service: CustomService
            max_items: int = 10
            
            def __init__(self, custom_service: CustomService, max_items: int = 10):
                super().__init__(
                    name="test_agent",
                    custom_service=custom_service,
                    max_items=max_items,
                )
        
        service = CustomService()
        agent = TestAgent(custom_service=service, max_items=5)
        
        assert agent.custom_service is service
        assert agent.max_items == 5


class TestAttributeModification:
    """Test if attributes can be modified after initialization."""
    
    def test_modify_after_init_direct_fails(self):
        """Test that modifying non-underscore attributes FAILS."""
        from google.adk.agents import BaseAgent
        
        class TestAgent(BaseAgent):
            def __init__(self, counter: int):
                super().__init__(name="test_agent")
                object.__setattr__(self, 'counter', counter)  # Use workaround
        
        agent = TestAgent(counter=0)
        assert agent.counter == 0
        
        # Modifying non-field attributes fails
        with pytest.raises(ValueError, match='has no field "counter"'):
            agent.counter = 10
    
    def test_modify_underscore_attribute_works(self):
        """Test that modifying underscore-prefixed attributes WORKS."""
        from google.adk.agents import BaseAgent
        
        class TestAgent(BaseAgent):
            def __init__(self, counter: int):
                super().__init__(name="test_agent")
                self._counter = counter
        
        agent = TestAgent(counter=0)
        assert agent._counter == 0
        
        # Modifying underscore-prefixed attributes works!
        agent._counter = 10
        assert agent._counter == 10
    
    def test_add_new_attribute_after_init_fails(self):
        """Test that adding new non-underscore attributes FAILS."""
        from google.adk.agents import BaseAgent
        
        class TestAgent(BaseAgent):
            def __init__(self):
                super().__init__(name="test_agent")
        
        agent = TestAgent()
        
        with pytest.raises(ValueError, match='has no field "new_attr"'):
            agent.new_attr = "new_value"
    
    def test_add_underscore_attribute_after_init_works(self):
        """Test that adding underscore-prefixed attributes WORKS."""
        from google.adk.agents import BaseAgent
        
        class TestAgent(BaseAgent):
            def __init__(self):
                super().__init__(name="test_agent")
        
        agent = TestAgent()
        
        # Adding underscore-prefixed attributes works!
        agent._new_attr = "new_value"
        assert agent._new_attr == "new_value"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

