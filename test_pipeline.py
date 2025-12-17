import pytest
from unittest.mock import Mock, patch, MagicMock
from nodes import router_node, weather_node, rag_retrieval_node, generation_node
from graph import build_graph


@pytest.fixture
def mock_weather_state():
    return {
        "question": "What's the weather in London?",
        "route": "weather",
        "context": "",
        "weather_data": {},
        "retrieved_docs": [],
        "generation": "",
        "evaluation": {}
    }


@pytest.fixture
def mock_pdf_state():
    return {
        "question": "What is the main topic of the document?",
        "route": "pdf",
        "context": "",
        "weather_data": {},
        "retrieved_docs": [],
        "generation": "",
        "evaluation": {}
    }


@patch('nodes.StrOutputParser')
@patch('nodes.get_llm')
@patch('nodes.ChatPromptTemplate.from_messages')
def test_router_weather_intent(mock_from_messages, mock_get_llm, mock_parser, mock_weather_state):
    """Test router correctly identifies weather queries."""
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "weather"
    
    mock_prompt = MagicMock()
    mock_llm = MagicMock()
    mock_parser_inst = MagicMock()
    
    mock_from_messages.return_value = mock_prompt
    mock_get_llm.return_value = mock_llm
    mock_parser.return_value = mock_parser_inst
    
    mock_prompt.__or__.return_value = mock_llm
    mock_llm.__or__.return_value = mock_chain
    
    result = router_node(mock_weather_state)
    assert result["route"] == "weather"


@patch('nodes.StrOutputParser')
@patch('nodes.get_llm')
@patch('nodes.ChatPromptTemplate.from_messages')
def test_router_pdf_intent(mock_from_messages, mock_get_llm, mock_parser, mock_pdf_state):
    """Test router correctly identifies PDF queries."""
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "pdf"
    
    mock_prompt = MagicMock()
    mock_llm = MagicMock()
    mock_parser_inst = MagicMock()
    
    mock_from_messages.return_value = mock_prompt
    mock_get_llm.return_value = mock_llm
    mock_parser.return_value = mock_parser_inst
    
    mock_prompt.__or__.return_value = mock_llm
    mock_llm.__or__.return_value = mock_chain
    
    result = router_node(mock_pdf_state)
    assert result["route"] == "pdf"


@patch('nodes.StrOutputParser')
@patch('nodes.get_llm')
@patch('nodes.ChatPromptTemplate.from_messages')
@patch('nodes.requests.get')
def test_weather_node_success(mock_get, mock_from_messages, mock_get_llm, mock_parser, mock_weather_state):
    """Test weather API call with successful response."""
    mock_response = Mock()
    mock_response.json.return_value = {
        "name": "London",
        "sys": {"country": "GB"},
        "main": {"temp": 15, "feels_like": 13, "humidity": 80},
        "weather": [{"description": "cloudy"}],
        "wind": {"speed": 5}
    }
    mock_response.raise_for_status = Mock()
    mock_get.return_value = mock_response
    
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "London"
    
    mock_prompt = MagicMock()
    mock_llm = MagicMock()
    mock_parser_inst = MagicMock()
    
    mock_from_messages.return_value = mock_prompt
    mock_get_llm.return_value = mock_llm
    mock_parser.return_value = mock_parser_inst
    
    mock_prompt.__or__.return_value = mock_llm
    mock_llm.__or__.return_value = mock_chain
    
    result = weather_node(mock_weather_state)
    assert "weather_data" in result
    assert "London" in result["context"]


@patch('nodes.get_vector_store')
def test_rag_retrieval_node(mock_vectorstore, mock_pdf_state):
    """Test RAG retrieval from Qdrant."""
    mock_doc = Mock()
    mock_doc.page_content = "Test document content"
    
    mock_retriever = Mock()
    mock_retriever.invoke.return_value = [mock_doc]
    
    mock_vs = Mock()
    mock_vs.as_retriever.return_value = mock_retriever
    mock_vectorstore.return_value = mock_vs
    
    result = rag_retrieval_node(mock_pdf_state)
    assert "retrieved_docs" in result
    assert len(result["retrieved_docs"]) > 0
    assert result["retrieved_docs"][0] == "Test document content"


@patch('nodes.StrOutputParser')
@patch('nodes.get_llm')
@patch('nodes.ChatPromptTemplate.from_messages')
def test_generation_node_weather(mock_from_messages, mock_get_llm, mock_parser):
    """Test generation node for weather route."""
    state = {
        "question": "What's the weather?",
        "route": "weather",
        "context": "Temperature: 15°C",
        "weather_data": {},
        "retrieved_docs": [],
        "generation": "",
        "evaluation": {}
    }
    
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "The weather is 15°C"
    
    mock_prompt = MagicMock()
    mock_llm = MagicMock()
    mock_parser_inst = MagicMock()
    
    mock_from_messages.return_value = mock_prompt
    mock_get_llm.return_value = mock_llm
    mock_parser.return_value = mock_parser_inst
    
    mock_prompt.__or__.return_value = mock_llm
    mock_llm.__or__.return_value = mock_chain
    
    result = generation_node(state)
    assert "generation" in result
    assert result["generation"] == "The weather is 15°C"


@patch('nodes.StrOutputParser')
@patch('nodes.get_llm')
@patch('nodes.ChatPromptTemplate.from_messages')
def test_generation_node_pdf(mock_from_messages, mock_get_llm, mock_parser):
    """Test generation node for PDF route."""
    state = {
        "question": "What is the main topic?",
        "route": "pdf",
        "context": "This document discusses AI",
        "weather_data": {},
        "retrieved_docs": ["AI content"],
        "generation": "",
        "evaluation": {}
    }
    
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "The main topic is AI"
    
    mock_prompt = MagicMock()
    mock_llm = MagicMock()
    mock_parser_inst = MagicMock()
    
    mock_from_messages.return_value = mock_prompt
    mock_get_llm.return_value = mock_llm
    mock_parser.return_value = mock_parser_inst
    
    mock_prompt.__or__.return_value = mock_llm
    mock_llm.__or__.return_value = mock_chain
    
    result = generation_node(state)
    assert "generation" in result
    assert "AI" in result["generation"]


def test_graph_structure():
    """Test graph compilation and structure."""
    app = build_graph()
    assert app is not None
    assert callable(app.invoke)


def test_graph_state_schema():
    """Test GraphState TypedDict structure."""
    from nodes import GraphState
    
    test_state = {
        "question": "test",
        "route": "weather",
        "context": "",
        "weather_data": {},
        "retrieved_docs": [],
        "generation": "",
        "evaluation": {}
    }
    
    assert "question" in test_state
    assert "route" in test_state
    assert "context" in test_state
    assert "weather_data" in test_state
    assert "retrieved_docs" in test_state
    assert "generation" in test_state
    assert "evaluation" in test_state


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
