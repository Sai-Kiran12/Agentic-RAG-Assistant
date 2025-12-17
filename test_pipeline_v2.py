import pytest
from unittest.mock import Mock, patch, MagicMock
from nodes import (
    router_node, 
    weather_node, 
    rag_retrieval_node, 
    generation_node,
    evaluation_node,
    GraphState
)
from graph import build_graph
import os


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_kalam_documents():
    """Mock documents about APJ Abdul Kalam."""
    return [
        "Dr. APJ Abdul Kalam, known as the Missile Man of India, was born on October 15, 1931 in Rameswaram, Tamil Nadu.",
        "Kalam served as the 11th President of India from 2002 to 2007. He was known as the People's President.",
        "Dr. Kalam played a pivotal role in India's Pokhran-II nuclear tests in 1998 and the development of India's missile program.",
        "He received the Bharat Ratna, India's highest civilian honor, in 1997 for his contributions to science and technology.",
        "Kalam was a scientist at ISRO and DRDO, contributing to India's civilian space program and military missile development.",
        "His famous books include 'Wings of Fire', 'India 2020', and 'Ignited Minds'.",
        "Dr. Kalam passed away on July 27, 2015, while delivering a lecture at IIM Shillong.",
        "He was known for his humility, dedication to education, and inspiring millions of young Indians.",
        "Kalam's work on the SLV-III project at ISRO made India capable of launching satellites.",
        "The Pokhran nuclear tests were conducted under Kalam's scientific leadership as Chief Scientific Adviser to the Prime Minister."
    ]


# ============================================================================
# 10 RAG TESTS FOR APJ ABDUL KALAM BIOGRAPHY
# ============================================================================

@patch('cohere.Client')  # 👈 Patch at cohere module level, not nodes.cohere
@patch('nodes.get_vector_store')
def test_rag_01_basic_kalam_query(mock_vectorstore, mock_cohere, mock_kalam_documents):
    """Test 1: Basic RAG retrieval for APJ Kalam biography."""
    state = {
        "question": "Who was APJ Abdul Kalam?",
        "route": "pdf",
        "context": "",
        "weather_data": {},
        "retrieved_docs": [],
        "rerank_scores": [],
        "generation": "",
        "evaluation": {}
    }
    
    # Mock Qdrant retrieval
    mock_docs = [Mock(page_content=doc) for doc in mock_kalam_documents]
    mock_retriever = Mock()
    mock_retriever.invoke.return_value = mock_docs
    mock_vs = Mock()
    mock_vs.as_retriever.return_value = mock_retriever
    mock_vectorstore.return_value = mock_vs
    
    # Mock Cohere reranking
    mock_cohere_client = Mock()
    mock_results = []
    for i, doc in enumerate(mock_kalam_documents[:4]):
        result = Mock()
        result.document = Mock(text=doc)
        result.relevance_score = 0.95 - (i * 0.05)
        mock_results.append(result)
    
    mock_rerank_response = Mock(results=mock_results)
    mock_cohere_client.rerank.return_value = mock_rerank_response
    mock_cohere.return_value = mock_cohere_client
    
    result = rag_retrieval_node(state)
    
    assert len(result["retrieved_docs"]) == 4
    assert len(result["rerank_scores"]) == 4
    assert "Kalam" in result["context"]


@patch('cohere.Client')
@patch('nodes.get_vector_store')
def test_rag_02_missile_man_query(mock_vectorstore, mock_cohere, mock_kalam_documents):
    """Test 2: Query about Kalam as Missile Man."""
    state = {
        "question": "Why is Kalam called the Missile Man of India?",
        "route": "pdf",
        "context": "",
        "weather_data": {},
        "retrieved_docs": [],
        "rerank_scores": [],
        "generation": "",
        "evaluation": {}
    }
    
    mock_docs = [Mock(page_content=doc) for doc in mock_kalam_documents]
    mock_retriever = Mock()
    mock_retriever.invoke.return_value = mock_docs
    mock_vs = Mock()
    mock_vs.as_retriever.return_value = mock_retriever
    mock_vectorstore.return_value = mock_vs
    
    mock_cohere_client = Mock()
    mock_results = [Mock(document=Mock(text=mock_kalam_documents[2]), relevance_score=0.98),
                   Mock(document=Mock(text=mock_kalam_documents[4]), relevance_score=0.94),
                   Mock(document=Mock(text=mock_kalam_documents[9]), relevance_score=0.91),
                   Mock(document=Mock(text=mock_kalam_documents[0]), relevance_score=0.87)]
    mock_rerank_response = Mock(results=mock_results)
    mock_cohere_client.rerank.return_value = mock_rerank_response
    mock_cohere.return_value = mock_cohere_client
    
    result = rag_retrieval_node(state)
    
    assert len(result["retrieved_docs"]) == 4
    assert "missile" in result["context"].lower()


@patch('cohere.Client')
@patch('nodes.get_vector_store')
def test_rag_03_president_query(mock_vectorstore, mock_cohere, mock_kalam_documents):
    """Test 3: Query about Kalam's presidency."""
    state = {
        "question": "When did APJ Abdul Kalam serve as President?",
        "route": "pdf",
        "context": "",
        "weather_data": {},
        "retrieved_docs": [],
        "rerank_scores": [],
        "generation": "",
        "evaluation": {}
    }
    
    mock_docs = [Mock(page_content=doc) for doc in mock_kalam_documents]
    mock_retriever = Mock()
    mock_retriever.invoke.return_value = mock_docs
    mock_vs = Mock()
    mock_vs.as_retriever.return_value = mock_retriever
    mock_vectorstore.return_value = mock_vs
    
    mock_cohere_client = Mock()
    mock_results = [Mock(document=Mock(text=doc), relevance_score=0.9-i*0.05) 
                   for i, doc in enumerate(mock_kalam_documents[:4])]
    mock_rerank_response = Mock(results=mock_results)
    mock_cohere_client.rerank.return_value = mock_rerank_response
    mock_cohere.return_value = mock_cohere_client
    
    result = rag_retrieval_node(state)
    
    assert "2002" in result["context"] or "President" in result["context"]


@patch('cohere.Client')
@patch('nodes.get_vector_store')
def test_rag_04_books_query(mock_vectorstore, mock_cohere, mock_kalam_documents):
    """Test 4: Query about Kalam's books."""
    state = {
        "question": "What books did Dr. Kalam write?",
        "route": "pdf",
        "context": "",
        "weather_data": {},
        "retrieved_docs": [],
        "rerank_scores": [],
        "generation": "",
        "evaluation": {}
    }
    
    mock_docs = [Mock(page_content=doc) for doc in mock_kalam_documents]
    mock_retriever = Mock()
    mock_retriever.invoke.return_value = mock_docs
    mock_vs = Mock()
    mock_vs.as_retriever.return_value = mock_retriever
    mock_vectorstore.return_value = mock_vs
    
    mock_cohere_client = Mock()
    mock_results = [Mock(document=Mock(text=mock_kalam_documents[5]), relevance_score=0.96),
                   Mock(document=Mock(text=mock_kalam_documents[0]), relevance_score=0.88),
                   Mock(document=Mock(text=mock_kalam_documents[1]), relevance_score=0.85),
                   Mock(document=Mock(text=mock_kalam_documents[7]), relevance_score=0.82)]
    mock_rerank_response = Mock(results=mock_results)
    mock_cohere_client.rerank.return_value = mock_rerank_response
    mock_cohere.return_value = mock_cohere_client
    
    result = rag_retrieval_node(state)
    
    assert "Wings of Fire" in result["context"] or "books" in result["context"].lower()


@patch('cohere.Client')
@patch('nodes.get_vector_store')
def test_rag_05_bharat_ratna_query(mock_vectorstore, mock_cohere, mock_kalam_documents):
    """Test 5: Query about Bharat Ratna award."""
    state = {
        "question": "When did Kalam receive the Bharat Ratna?",
        "route": "pdf",
        "context": "",
        "weather_data": {},
        "retrieved_docs": [],
        "rerank_scores": [],
        "generation": "",
        "evaluation": {}
    }
    
    mock_docs = [Mock(page_content=doc) for doc in mock_kalam_documents]
    mock_retriever = Mock()
    mock_retriever.invoke.return_value = mock_docs
    mock_vs = Mock()
    mock_vs.as_retriever.return_value = mock_retriever
    mock_vectorstore.return_value = mock_vs
    
    mock_cohere_client = Mock()
    mock_results = [Mock(document=Mock(text=doc), relevance_score=0.93-i*0.04) 
                   for i, doc in enumerate(mock_kalam_documents[:4])]
    mock_rerank_response = Mock(results=mock_results)
    mock_cohere_client.rerank.return_value = mock_rerank_response
    mock_cohere.return_value = mock_cohere_client
    
    result = rag_retrieval_node(state)
    
    assert len(result["retrieved_docs"]) == 4
    assert result["rerank_scores"][0] > 0.8


@patch('cohere.Client')
@patch('nodes.get_vector_store')
def test_rag_06_birthplace_query(mock_vectorstore, mock_cohere, mock_kalam_documents):
    """Test 6: Query about Kalam's birthplace."""
    state = {
        "question": "Where was APJ Abdul Kalam born?",
        "route": "pdf",
        "context": "",
        "weather_data": {},
        "retrieved_docs": [],
        "rerank_scores": [],
        "generation": "",
        "evaluation": {}
    }
    
    mock_docs = [Mock(page_content=doc) for doc in mock_kalam_documents]
    mock_retriever = Mock()
    mock_retriever.invoke.return_value = mock_docs
    mock_vs = Mock()
    mock_vs.as_retriever.return_value = mock_retriever
    mock_vectorstore.return_value = mock_vs
    
    mock_cohere_client = Mock()
    mock_results = [Mock(document=Mock(text=doc), relevance_score=0.97-i*0.03) 
                   for i, doc in enumerate(mock_kalam_documents[:4])]
    mock_rerank_response = Mock(results=mock_results)
    mock_cohere_client.rerank.return_value = mock_rerank_response
    mock_cohere.return_value = mock_cohere_client
    
    result = rag_retrieval_node(state)
    
    assert "Rameswaram" in result["context"] or "Tamil Nadu" in result["context"]


@patch('cohere.Client')
@patch('nodes.get_vector_store')
def test_rag_07_death_query(mock_vectorstore, mock_cohere, mock_kalam_documents):
    """Test 7: Query about Kalam's death."""
    state = {
        "question": "When and how did Dr. Kalam pass away?",
        "route": "pdf",
        "context": "",
        "weather_data": {},
        "retrieved_docs": [],
        "rerank_scores": [],
        "generation": "",
        "evaluation": {}
    }
    
    mock_docs = [Mock(page_content=doc) for doc in mock_kalam_documents]
    mock_retriever = Mock()
    mock_retriever.invoke.return_value = mock_docs
    mock_vs = Mock()
    mock_vs.as_retriever.return_value = mock_retriever
    mock_vectorstore.return_value = mock_vs
    
    mock_cohere_client = Mock()
    # Specifically return document with death information first
    mock_results = [
        Mock(document=Mock(text=mock_kalam_documents[6]), relevance_score=0.98),  # "Dr. Kalam passed away on July 27, 2015..."
        Mock(document=Mock(text=mock_kalam_documents[1]), relevance_score=0.88),
        Mock(document=Mock(text=mock_kalam_documents[0]), relevance_score=0.85),
        Mock(document=Mock(text=mock_kalam_documents[7]), relevance_score=0.82)
    ]
    mock_rerank_response = Mock(results=mock_results)
    mock_cohere_client.rerank.return_value = mock_rerank_response
    mock_cohere.return_value = mock_cohere_client
    
    result = rag_retrieval_node(state)
    
    # Verify death information is in context
    assert "2015" in result["context"]
    assert len(result["retrieved_docs"]) == 4


@patch('cohere.Client')
@patch('nodes.get_vector_store')
def test_rag_08_isro_drdo_query(mock_vectorstore, mock_cohere, mock_kalam_documents):
    """Test 8: Query about ISRO and DRDO work."""
    state = {
        "question": "What was Kalam's contribution to ISRO and DRDO?",
        "route": "pdf",
        "context": "",
        "weather_data": {},
        "retrieved_docs": [],
        "rerank_scores": [],
        "generation": "",
        "evaluation": {}
    }
    
    mock_docs = [Mock(page_content=doc) for doc in mock_kalam_documents]
    mock_retriever = Mock()
    mock_retriever.invoke.return_value = mock_docs
    mock_vs = Mock()
    mock_vs.as_retriever.return_value = mock_retriever
    mock_vectorstore.return_value = mock_vs
    
    mock_cohere_client = Mock()
    mock_results = [Mock(document=Mock(text=doc), relevance_score=0.95-i*0.04) 
                   for i, doc in enumerate(mock_kalam_documents[:4])]
    mock_rerank_response = Mock(results=mock_results)
    mock_cohere_client.rerank.return_value = mock_rerank_response
    mock_cohere.return_value = mock_cohere_client
    
    result = rag_retrieval_node(state)
    
    assert len(result["retrieved_docs"]) > 0
    assert all(score > 0 for score in result["rerank_scores"])


@patch('cohere.Client')
@patch('nodes.get_vector_store')
def test_rag_09_rerank_scores_validation(mock_vectorstore, mock_cohere, mock_kalam_documents):
    """Test 9: Validate reranking scores are properly returned."""
    state = {
        "question": "Tell me about APJ Abdul Kalam",
        "route": "pdf",
        "context": "",
        "weather_data": {},
        "retrieved_docs": [],
        "rerank_scores": [],
        "generation": "",
        "evaluation": {}
    }
    
    mock_docs = [Mock(page_content=doc) for doc in mock_kalam_documents]
    mock_retriever = Mock()
    mock_retriever.invoke.return_value = mock_docs
    mock_vs = Mock()
    mock_vs.as_retriever.return_value = mock_retriever
    mock_vectorstore.return_value = mock_vs
    
    mock_cohere_client = Mock()
    mock_results = [Mock(document=Mock(text=doc), relevance_score=0.99-i*0.02) 
                   for i, doc in enumerate(mock_kalam_documents[:4])]
    mock_rerank_response = Mock(results=mock_results)
    mock_cohere_client.rerank.return_value = mock_rerank_response
    mock_cohere.return_value = mock_cohere_client
    
    result = rag_retrieval_node(state)
    
    # Validate scores are in descending order
    assert result["rerank_scores"] == sorted(result["rerank_scores"], reverse=True)
    # Validate scores are between 0 and 1
    assert all(0 <= score <= 1 for score in result["rerank_scores"])
    # Validate we got 4 results
    assert len(result["rerank_scores"]) == 4


@patch('nodes.get_vector_store')
def test_rag_10_error_handling(mock_vectorstore):
    """Test 10: RAG error handling when vector store fails."""
    state = {
        "question": "Who was Kalam?",
        "route": "pdf",
        "context": "",
        "weather_data": {},
        "retrieved_docs": [],
        "rerank_scores": [],
        "generation": "",
        "evaluation": {}
    }
    
    mock_vectorstore.side_effect = Exception("Qdrant connection failed")
    
    result = rag_retrieval_node(state)
    
    assert result["retrieved_docs"] == []
    assert result["rerank_scores"] == []
    assert "Error" in result["context"]


# ============================================================================
# 10 WEATHER API TESTS (Keep the same - they work fine)
# ============================================================================

@patch('nodes.StrOutputParser')
@patch('nodes.get_llm')
@patch('nodes.ChatPromptTemplate.from_messages')
@patch('nodes.requests.get')
def test_weather_01_mumbai_success(mock_get, mock_from_messages, mock_get_llm, mock_parser):
    """Test 1: Successful weather query for Mumbai."""
    state = {
        "question": "What's the weather in Mumbai?",
        "route": "weather",
        "context": "",
        "weather_data": {},
        "retrieved_docs": [],
        "rerank_scores": [],
        "generation": "",
        "evaluation": {}
    }
    
    mock_response = Mock()
    mock_response.json.return_value = {
        "name": "Mumbai",
        "sys": {"country": "IN"},
        "main": {"temp": 32, "feels_like": 35, "humidity": 70},
        "weather": [{"description": "haze"}],
        "wind": {"speed": 4.2}
    }
    mock_response.raise_for_status = Mock()
    mock_get.return_value = mock_response
    
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "Mumbai"
    mock_prompt = MagicMock()
    mock_llm = MagicMock()
    mock_parser_inst = MagicMock()
    mock_from_messages.return_value = mock_prompt
    mock_get_llm.return_value = mock_llm
    mock_parser.return_value = mock_parser_inst
    mock_prompt.__or__.return_value = mock_llm
    mock_llm.__or__.return_value = mock_chain
    
    result = weather_node(state)
    
    assert "Mumbai" in result["context"]
    assert "32" in result["context"]
    assert result["weather_data"]["name"] == "Mumbai"


@patch('nodes.StrOutputParser')
@patch('nodes.get_llm')
@patch('nodes.ChatPromptTemplate.from_messages')
@patch('nodes.requests.get')
def test_weather_02_delhi_success(mock_get, mock_from_messages, mock_get_llm, mock_parser):
    """Test 2: Successful weather query for Delhi."""
    state = {
        "question": "How's the weather in Delhi today?",
        "route": "weather",
        "context": "",
        "weather_data": {},
        "retrieved_docs": [],
        "rerank_scores": [],
        "generation": "",
        "evaluation": {}
    }
    
    mock_response = Mock()
    mock_response.json.return_value = {
        "name": "Delhi",
        "sys": {"country": "IN"},
        "main": {"temp": 18, "feels_like": 16, "humidity": 45},
        "weather": [{"description": "clear sky"}],
        "wind": {"speed": 3.1}
    }
    mock_response.raise_for_status = Mock()
    mock_get.return_value = mock_response
    
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "Delhi"
    mock_prompt = MagicMock()
    mock_llm = MagicMock()
    mock_parser_inst = MagicMock()
    mock_from_messages.return_value = mock_prompt
    mock_get_llm.return_value = mock_llm
    mock_parser.return_value = mock_parser_inst
    mock_prompt.__or__.return_value = mock_llm
    mock_llm.__or__.return_value = mock_chain
    
    result = weather_node(state)
    
    assert "Delhi" in result["context"]
    assert "18" in result["context"]


@patch('nodes.StrOutputParser')
@patch('nodes.get_llm')
@patch('nodes.ChatPromptTemplate.from_messages')
@patch('nodes.requests.get')
def test_weather_03_bangalore_success(mock_get, mock_from_messages, mock_get_llm, mock_parser):
    """Test 3: Successful weather query for Bangalore."""
    state = {
        "question": "Tell me the weather in Bangalore",
        "route": "weather",
        "context": "",
        "weather_data": {},
        "retrieved_docs": [],
        "rerank_scores": [],
        "generation": "",
        "evaluation": {}
    }
    
    mock_response = Mock()
    mock_response.json.return_value = {
        "name": "Bangalore",
        "sys": {"country": "IN"},
        "main": {"temp": 24, "feels_like": 23, "humidity": 60},
        "weather": [{"description": "partly cloudy"}],
        "wind": {"speed": 2.5}
    }
    mock_response.raise_for_status = Mock()
    mock_get.return_value = mock_response
    
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "Bangalore"
    mock_prompt = MagicMock()
    mock_llm = MagicMock()
    mock_parser_inst = MagicMock()
    mock_from_messages.return_value = mock_prompt
    mock_get_llm.return_value = mock_llm
    mock_parser.return_value = mock_parser_inst
    mock_prompt.__or__.return_value = mock_llm
    mock_llm.__or__.return_value = mock_chain
    
    result = weather_node(state)
    
    assert "Bangalore" in result["context"]
    assert "weather_data" in result


@patch('nodes.StrOutputParser')
@patch('nodes.get_llm')
@patch('nodes.ChatPromptTemplate.from_messages')
@patch('nodes.requests.get')
def test_weather_04_kolkata_humidity(mock_get, mock_from_messages, mock_get_llm, mock_parser):
    """Test 4: Weather query for Kolkata with humidity check."""
    state = {
        "question": "What's the humidity in Kolkata?",
        "route": "weather",
        "context": "",
        "weather_data": {},
        "retrieved_docs": [],
        "rerank_scores": [],
        "generation": "",
        "evaluation": {}
    }
    
    mock_response = Mock()
    mock_response.json.return_value = {
        "name": "Kolkata",
        "sys": {"country": "IN"},
        "main": {"temp": 29, "feels_like": 32, "humidity": 85},
        "weather": [{"description": "humid"}],
        "wind": {"speed": 2.8}
    }
    mock_response.raise_for_status = Mock()
    mock_get.return_value = mock_response
    
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "Kolkata"
    mock_prompt = MagicMock()
    mock_llm = MagicMock()
    mock_parser_inst = MagicMock()
    mock_from_messages.return_value = mock_prompt
    mock_get_llm.return_value = mock_llm
    mock_parser.return_value = mock_parser_inst
    mock_prompt.__or__.return_value = mock_llm
    mock_llm.__or__.return_value = mock_chain
    
    result = weather_node(state)
    
    assert "85" in result["context"]
    assert "Humidity" in result["context"]


@patch('nodes.StrOutputParser')
@patch('nodes.get_llm')
@patch('nodes.ChatPromptTemplate.from_messages')
@patch('nodes.requests.get')
def test_weather_05_chennai_temperature(mock_get, mock_from_messages, mock_get_llm, mock_parser):
    """Test 5: Weather query for Chennai with temperature validation."""
    state = {
        "question": "What's the temperature in Chennai?",
        "route": "weather",
        "context": "",
        "weather_data": {},
        "retrieved_docs": [],
        "rerank_scores": [],
        "generation": "",
        "evaluation": {}
    }
    
    mock_response = Mock()
    mock_response.json.return_value = {
        "name": "Chennai",
        "sys": {"country": "IN"},
        "main": {"temp": 31, "feels_like": 34, "humidity": 75},
        "weather": [{"description": "sunny"}],
        "wind": {"speed": 5.5}
    }
    mock_response.raise_for_status = Mock()
    mock_get.return_value = mock_response
    
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "Chennai"
    mock_prompt = MagicMock()
    mock_llm = MagicMock()
    mock_parser_inst = MagicMock()
    mock_from_messages.return_value = mock_prompt
    mock_get_llm.return_value = mock_llm
    mock_parser.return_value = mock_parser_inst
    mock_prompt.__or__.return_value = mock_llm
    mock_llm.__or__.return_value = mock_chain
    
    result = weather_node(state)
    
    assert "31" in result["context"]
    assert "Temperature" in result["context"]


@patch('nodes.StrOutputParser')
@patch('nodes.get_llm')
@patch('nodes.ChatPromptTemplate.from_messages')
@patch('nodes.requests.get')
def test_weather_06_api_timeout_error(mock_get, mock_from_messages, mock_get_llm, mock_parser):
    """Test 6: Handle API timeout error."""
    state = {
        "question": "Weather in Pune?",
        "route": "weather",
        "context": "",
        "weather_data": {},
        "retrieved_docs": [],
        "rerank_scores": [],
        "generation": "",
        "evaluation": {}
    }
    
    mock_get.side_effect = Exception("Timeout error")
    
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "Pune"
    mock_prompt = MagicMock()
    mock_llm = MagicMock()
    mock_parser_inst = MagicMock()
    mock_from_messages.return_value = mock_prompt
    mock_get_llm.return_value = mock_llm
    mock_parser.return_value = mock_parser_inst
    mock_prompt.__or__.return_value = mock_llm
    mock_llm.__or__.return_value = mock_chain
    
    result = weather_node(state)
    
    assert "Error" in result["context"]
    assert result["weather_data"] == {}


@patch('nodes.StrOutputParser')
@patch('nodes.get_llm')
@patch('nodes.ChatPromptTemplate.from_messages')
@patch('nodes.requests.get')
def test_weather_07_invalid_city_error(mock_get, mock_from_messages, mock_get_llm, mock_parser):
    """Test 7: Handle invalid city name error."""
    state = {
        "question": "Weather in InvalidCity123?",
        "route": "weather",
        "context": "",
        "weather_data": {},
        "retrieved_docs": [],
        "rerank_scores": [],
        "generation": "",
        "evaluation": {}
    }
    
    mock_response = Mock()
    mock_response.raise_for_status.side_effect = Exception("City not found")
    mock_get.return_value = mock_response
    
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "InvalidCity123"
    mock_prompt = MagicMock()
    mock_llm = MagicMock()
    mock_parser_inst = MagicMock()
    mock_from_messages.return_value = mock_prompt
    mock_get_llm.return_value = mock_llm
    mock_parser.return_value = mock_parser_inst
    mock_prompt.__or__.return_value = mock_llm
    mock_llm.__or__.return_value = mock_chain
    
    result = weather_node(state)
    
    assert "Error" in result["context"]


@patch('nodes.StrOutputParser')
@patch('nodes.get_llm')
@patch('nodes.ChatPromptTemplate.from_messages')
@patch('nodes.requests.get')
def test_weather_08_wind_speed_check(mock_get, mock_from_messages, mock_get_llm, mock_parser):
    """Test 8: Weather query with wind speed validation."""
    state = {
        "question": "What's the wind speed in Hyderabad?",
        "route": "weather",
        "context": "",
        "weather_data": {},
        "retrieved_docs": [],
        "rerank_scores": [],
        "generation": "",
        "evaluation": {}
    }
    
    mock_response = Mock()
    mock_response.json.return_value = {
        "name": "Hyderabad",
        "sys": {"country": "IN"},
        "main": {"temp": 27, "feels_like": 28, "humidity": 55},
        "weather": [{"description": "windy"}],
        "wind": {"speed": 8.5}
    }
    mock_response.raise_for_status = Mock()
    mock_get.return_value = mock_response
    
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "Hyderabad"
    mock_prompt = MagicMock()
    mock_llm = MagicMock()
    mock_parser_inst = MagicMock()
    mock_from_messages.return_value = mock_prompt
    mock_get_llm.return_value = mock_llm
    mock_parser.return_value = mock_parser_inst
    mock_prompt.__or__.return_value = mock_llm
    mock_llm.__or__.return_value = mock_chain
    
    result = weather_node(state)
    
    assert "8.5" in result["context"]
    assert "Wind Speed" in result["context"]


@patch('nodes.StrOutputParser')
@patch('nodes.get_llm')
@patch('nodes.ChatPromptTemplate.from_messages')
@patch('nodes.requests.get')
def test_weather_09_feels_like_temperature(mock_get, mock_from_messages, mock_get_llm, mock_parser):
    """Test 9: Weather query with 'feels like' temperature."""
    state = {
        "question": "How does it feel in Ahmedabad?",
        "route": "weather",
        "context": "",
        "weather_data": {},
        "retrieved_docs": [],
        "rerank_scores": [],
        "generation": "",
        "evaluation": {}
    }
    
    mock_response = Mock()
    mock_response.json.return_value = {
        "name": "Ahmedabad",
        "sys": {"country": "IN"},
        "main": {"temp": 35, "feels_like": 38, "humidity": 40},
        "weather": [{"description": "hot"}],
        "wind": {"speed": 4.0}
    }
    mock_response.raise_for_status = Mock()
    mock_get.return_value = mock_response
    
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "Ahmedabad"
    mock_prompt = MagicMock()
    mock_llm = MagicMock()
    mock_parser_inst = MagicMock()
    mock_from_messages.return_value = mock_prompt
    mock_get_llm.return_value = mock_llm
    mock_parser.return_value = mock_parser_inst
    mock_prompt.__or__.return_value = mock_llm
    mock_llm.__or__.return_value = mock_chain
    
    result = weather_node(state)
    
    assert "38" in result["context"]
    assert "Feels Like" in result["context"]


@patch('nodes.StrOutputParser')
@patch('nodes.get_llm')
@patch('nodes.ChatPromptTemplate.from_messages')
@patch('nodes.requests.get')
def test_weather_10_response_structure_validation(mock_get, mock_from_messages, mock_get_llm, mock_parser):
    """Test 10: Validate complete weather response structure."""
    state = {
        "question": "Weather report for Jaipur",
        "route": "weather",
        "context": "",
        "weather_data": {},
        "retrieved_docs": [],
        "rerank_scores": [],
        "generation": "",
        "evaluation": {}
    }
    
    mock_response = Mock()
    mock_response.json.return_value = {
        "name": "Jaipur",
        "sys": {"country": "IN"},
        "main": {"temp": 22, "feels_like": 20, "humidity": 50},
        "weather": [{"description": "pleasant"}],
        "wind": {"speed": 3.2}
    }
    mock_response.raise_for_status = Mock()
    mock_get.return_value = mock_response
    
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "Jaipur"
    mock_prompt = MagicMock()
    mock_llm = MagicMock()
    mock_parser_inst = MagicMock()
    mock_from_messages.return_value = mock_prompt
    mock_get_llm.return_value = mock_llm
    mock_parser.return_value = mock_parser_inst
    mock_prompt.__or__.return_value = mock_llm
    mock_llm.__or__.return_value = mock_chain
    
    result = weather_node(state)
    
    # Validate all required fields are present
    assert "context" in result
    assert "weather_data" in result
    assert "Location:" in result["context"]
    assert "Temperature:" in result["context"]
    assert "Feels Like:" in result["context"]
    assert "Humidity:" in result["context"]
    assert "Weather:" in result["context"]
    assert "Wind Speed:" in result["context"]


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])