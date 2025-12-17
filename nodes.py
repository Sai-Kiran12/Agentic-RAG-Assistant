import requests
from typing import TypedDict, Literal, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import (
    OPENWEATHER_API_KEY, 
    OPENWEATHER_BASE_URL, 
    OPENAI_API_KEY,
    COHERE_API_KEY,
    RETRIEVAL_TOP_K,
    RERANK_TOP_N
)
from utils import get_vector_store


# Define State
class GraphState(TypedDict):
    """State of the graph."""
    question: str
    route: Literal["weather", "pdf"]
    context: str
    weather_data: dict
    retrieved_docs: List[str]
    rerank_scores: List[float]
    generation: str
    evaluation: dict


# Initialize LLM lazily
def get_llm():
    """Get or create LLM instance."""
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=OPENAI_API_KEY
    )


def router_node(state: GraphState) -> GraphState:
    """Route query to weather API or PDF RAG based on intent."""
    question = state["question"]
    
    router_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a routing assistant. Analyze the user's question and determine if it's about:
        1. WEATHER: Questions about weather, temperature, climate, forecast for a location
        2. PDF: Questions about document content, information retrieval from documents
        
        Respond with ONLY one word: 'weather' or 'pdf'"""),
        ("human", "{question}")
    ])
    
    chain = router_prompt | get_llm() | StrOutputParser()
    route = chain.invoke({"question": question}).strip().lower()
    
    print(f"🔀 Router Decision: {route}")
    return {"route": route}


def weather_node(state: GraphState) -> GraphState:
    """Fetch weather data from OpenWeatherMap API."""
    question = state["question"]
    
    # Extract location using LLM
    location_prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract ONLY the city name from the user's question. Return only the city name, nothing else."),
        ("human", "{question}")
    ])
    
    chain = location_prompt | get_llm() | StrOutputParser()
    city = chain.invoke({"question": question}).strip()
    
    print(f"🌍 Fetching weather for: {city}")
    
    # Fetch weather data
    params = {
        "q": city,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric"
    }
    
    try:
        response = requests.get(OPENWEATHER_BASE_URL, params=params, timeout=10)
        response.raise_for_status()
        weather_data = response.json()
        
        context = f"""
        Location: {weather_data['name']}, {weather_data['sys']['country']}
        Temperature: {weather_data['main']['temp']}°C
        Feels Like: {weather_data['main']['feels_like']}°C
        Humidity: {weather_data['main']['humidity']}%
        Weather: {weather_data['weather'][0]['description']}
        Wind Speed: {weather_data['wind']['speed']} m/s
        """
        
        return {"weather_data": weather_data, "context": context}
    
    except Exception as e:
        error_context = f"Error fetching weather data: {str(e)}"
        print(f"❌ {error_context}")
        return {"weather_data": {}, "context": error_context}


def rag_retrieval_node(state: GraphState) -> GraphState:
    """Retrieve relevant documents from Qdrant with manual reranking."""
    question = state["question"]
    
    print(f"📚 Retrieving documents for: {question}")
    
    try:
        vectorstore = get_vector_store()
        
        # Step 1: Initial retrieval from Qdrant with higher k
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": RETRIEVAL_TOP_K}  # Get 10 candidates
        )
        
        initial_docs = retriever.invoke(question)
        print(f"🔍 Initial retrieval: {len(initial_docs)} candidates")
        
        # Step 2: Manual reranking with Cohere
        import cohere
        co = cohere.Client(api_key=COHERE_API_KEY)
        
        # Prepare documents for reranking
        documents = [doc.page_content for doc in initial_docs]
        
        # Rerank with Cohere
        rerank_response = co.rerank(
            model="rerank-english-v3.0",
            query=question,
            documents=documents,
            top_n=RERANK_TOP_N,
            return_documents=True
        )
        
        print(f"✨ Reranked to top {len(rerank_response.results)} documents")
        
        # Step 3: Extract reranked documents and scores
        retrieved_texts = []
        rerank_scores = []
        
        for result in rerank_response.results:
            retrieved_texts.append(result.document.text)
            rerank_scores.append(result.relevance_score)
        
        context = "\n\n".join(retrieved_texts)
        
        print(f"✅ Retrieved {len(retrieved_texts)} documents with scores: {[f'{s:.4f}' for s in rerank_scores]}")
        
        return {
            "retrieved_docs": retrieved_texts, 
            "context": context,
            "rerank_scores": rerank_scores
        }
    
    except Exception as e:
        error_context = f"Error retrieving documents: {str(e)}"
        print(f"❌ {error_context}")
        return {
            "retrieved_docs": [], 
            "context": error_context,
            "rerank_scores": []
        }


def generation_node(state: GraphState) -> GraphState:
    """Generate final response using LLM with context."""
    question = state["question"]
    context = state.get("context", "")
    route = state["route"]
    
    if route == "weather":
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful weather assistant. Use the provided weather data to answer the user's question naturally and conversationally."""),
            ("human", "Weather Data:\n{context}\n\nQuestion: {question}\n\nAnswer:")
        ])
    else:  # pdf
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant. Use the provided document context to answer the user's question. 
            If the answer is not in the context, say so. Be concise and accurate."""),
            ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:")
        ])
    
    chain = prompt | get_llm() | StrOutputParser()
    generation = chain.invoke({"context": context, "question": question})
    
    print(f"✨ Generated response")
    
    return {"generation": generation}


def evaluation_node(state: GraphState) -> GraphState:
    """Evaluate response quality using LangSmith criteria."""
    question = state["question"]
    generation = state["generation"]
    context = state.get("context", "")
    
    eval_prompt = ChatPromptTemplate.from_messages([
        ("system", """Evaluate the response on a scale of 1-10 for:
        1. Relevance: Does it answer the question?
        2. Accuracy: Is the information correct based on context?
        3. Completeness: Is the answer complete?
        
        You MUST respond with ONLY valid JSON in this exact format, no other text: 
        {{"relevance": X, "accuracy": Y, "completeness": Z}}"""),
        ("human", "Question: {question}\n\nContext: {context}\n\nResponse: {generation}")
    ])
    
    chain = eval_prompt | get_llm() | StrOutputParser()
    
    try:
        eval_result = chain.invoke({
            "question": question,
            "context": context,
            "generation": generation
        })

        print(f"🧪 Evaluation result: {eval_result}")

        # Clean the response (remove markdown code blocks if present)
        eval_result = eval_result.strip()
        if eval_result.startswith("```json"):
            eval_result = eval_result.replace("```json", "").replace("```", "")
        elif eval_result.startswith("```"):
            eval_result = eval_result.replace("```", "")
        
        import json
        evaluation = json.loads(eval_result)
        print(f"📊 Evaluation: {evaluation}")
        
        return {"evaluation": evaluation}
    
    except Exception as e:
        print(f"⚠️ Evaluation error: {e}")
        return {"evaluation": {"error": str(e)}}
