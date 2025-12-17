from langgraph.graph import StateGraph, START, END
from nodes import (
    GraphState, 
    router_node, 
    weather_node, 
    rag_retrieval_node, 
    generation_node,
    evaluation_node
)


def route_question(state: GraphState) -> str:
    """Conditional edge function to route based on router decision."""
    route = state.get("route", "pdf")
    if route == "weather":
        return "weather"
    else:
        return "pdf"


def build_graph():
    """Build and compile the LangGraph workflow."""
    # Initialize graph
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("weather", weather_node)
    workflow.add_node("rag_retrieval", rag_retrieval_node)
    workflow.add_node("generation", generation_node)
    workflow.add_node("evaluation", evaluation_node)
    
    # Add edges
    workflow.add_edge(START, "router")
    
    # Conditional edges from router
    workflow.add_conditional_edges(
        "router",
        route_question,
        {
            "weather": "weather",
            "pdf": "rag_retrieval"
        }
    )
    
    # Both paths lead to generation
    workflow.add_edge("weather", "generation")
    workflow.add_edge("rag_retrieval", "generation")
    
    # Generation leads to evaluation
    workflow.add_edge("generation", "evaluation")
    
    # Evaluation leads to END
    workflow.add_edge("evaluation", END)
    
    # Compile graph
    app = workflow.compile()
    
    return app


# Create graph instance
graph_app = build_graph()
