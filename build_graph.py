from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional, Dict, List, Any
from agents import PlannerAgent, GeneratorAgent, ReviewerAgent, ComparatorAgent
import json
from datetime import datetime
from dotenv import load_dotenv
import os
import sys
import yaml
import numpy as np

# Load environment variables
load_dotenv()


# ----- 1. State Definition -----
class ReviewState(TypedDict):
    total_reviews: int
    plan: List[Dict[str, Any]]
    current_index: int
    
    detailed_plan: Optional[Dict[str, Any]]  # New field for per-review detailed plan
    persona: Optional[Dict]
    rating: Optional[int]
    review: Optional[str]
    quality_assessment: Optional[Dict]
    attempt: int
    max_attempts: int
    
    all_synthetic_reviews: List[Dict]
    real_reviews: List[str]
    comparison_report: Optional[Dict]


# ----- 2. Initialization -----

def load_config(config_path: str = 'config/generation_config.yaml') -> Dict:
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}

config = load_config()
provider = os.getenv('LLM_PROVIDER', 'google').lower() # Renamed to LLM_PROVIDER as per user request

# Instantiate Agents
planner = PlannerAgent(config=config)
generator_agent = GeneratorAgent(config=config, provider=provider)
reviewer_agent = ReviewerAgent(config=config, provider=provider) 
comparator_agent = ComparatorAgent(config=config)


# ----- 3. Nodes -----

def planner_node(state: ReviewState) -> ReviewState:
    print(f"\n╔══════════════════════════════════════════════════════════════╗")
    print(f"║ 📋 GLOBAL PLANNER AGENT                                      ║")
    print(f"╠══════════════════════════════════════════════════════════════╣")
    print(f"║ Task: Batch Plan for {state['total_reviews']} reviews")
    
    plan = planner.plan_batch(state['total_reviews'])
    
    print(f"║ Plan Details:")
    for idx, item in enumerate(plan):
        print(f"║   {idx+1}. {item['rating']}⭐ - {item['persona']['name']}")
    print(f"╚══════════════════════════════════════════════════════════════╝")
    
    return {
        **state,
        "plan": plan,
        "current_index": 0,
        "all_synthetic_reviews": [],
        "detailed_plan": None
    }

def review_planner_node(state: ReviewState) -> ReviewState:
    """Detailed planner for a SINGLE review."""
    current_item = state['plan'][state['current_index']]
    
    print(f"\n╔══════════════════════════════════════════════════════════════╗")
    print(f"║ 📝 REVIEW PLANNER AGENT                                      ║")
    print(f"╠══════════════════════════════════════════════════════════════╣")
    
    feedback = None
    if state["attempt"] > 0:
        feedback = ", ".join(state["quality_assessment"].get("issues", []))
        print(f"║ ⚠️  RETRYING with Feedback: {feedback}")

    detailed_plan = planner.create_detailed_plan(
        rating=current_item['rating'],
        persona=current_item['persona'],
        feedback=feedback
    )
    
    print(f"║ 🗺️  Detailed Plan Created")
    print(f"╚══════════════════════════════════════════════════════════════╝")
    
    return {
        **state,
        "detailed_plan": detailed_plan
    }

def generator_node(state: ReviewState) -> ReviewState:
    plan = state['detailed_plan']
    
    print(f"\n┌──────────────────────────────────────────────────────────────┐")
    print(f"│ 🔄 GENERATOR AGENT (Executor)                                │")
    print(f"├──────────────────────────────────────────────────────────────┤")
    print(f"│ Item: {state['current_index'] + 1}/{state['total_reviews']} (Attempt {state['attempt'] + 1}/{state['max_attempts']})")
    
    review = generator_agent.execute_plan(
        plan=plan,
        real_reviews=state.get('real_reviews', [])
    )
    print(f"│ Generated: \"{review[:50]}...\"")
    print(f"└──────────────────────────────────────────────────────────────┘")
    
    return {
        **state,
        "review": review,
        "attempt": state["attempt"] + 1
    }

def reviewer_node(state: ReviewState) -> ReviewState:
    print(f"\n┌──────────────────────────────────────────────────────────────┐")
    print(f"│ 🔍 REVIEWER AGENT                                            │")
    print(f"├──────────────────────────────────────────────────────────────┤")
    
    assessment = reviewer_agent.review_review(
        review_text=state["review"],
        rating=state["detailed_plan"]["target_rating"],
        persona={"name": state["detailed_plan"]["persona_name"]} 
    )
    
    score = assessment.get('overall_score', 0)
    passed = assessment.get('pass', False)
    issues = assessment.get('issues', [])
    
    print(f"│ Quality Score: {score}/10")
    print(f"│ Verdict: {'✅ PASS' if passed else '❌ FAIL'}")
    if issues:
        print(f"│ Issues Found:")
        for issue in issues:
            print(f"│   - {issue}")
            
    print(f"└──────────────────────────────────────────────────────────────┘")
    return {**state, "quality_assessment": assessment}

def finalize_node(state: ReviewState) -> ReviewState:
    action = "✅ Accepted" if state["quality_assessment"].get("pass") else "⚠️ Accepted (Poor Quality)"
    print(f"\n[System]: Review {state['current_index'] + 1} finalized -> {action}")
    
    review_data = {
        "text": state["review"],
        "rating": state["detailed_plan"]["target_rating"],
        "persona": state["detailed_plan"]["persona_name"],
        "provider": generator_agent.get_name(),
        "attempts": state["attempt"],
        "quality_assessment": state["quality_assessment"],
        "generated_at": datetime.now().isoformat()
    }
    
    all_synthetic = state.get("all_synthetic_reviews", [])
    all_synthetic.append(review_data)
    
    return {
        **state,
        "all_synthetic_reviews": all_synthetic,
        "current_index": state["current_index"] + 1,
        "attempt": 0,
        "review": None,
        "detailed_plan": None,
        "quality_assessment": None
    }

async def run_mcp_tool(tool_name: str, arguments: dict):
    """Run a tool via MCP stdio client."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["tools/mcp_server.py"],
        env=env
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)
            return result.content[0].text

def comparator_node(state: ReviewState) -> ReviewState:
    print(f"\n╔══════════════════════════════════════════════════════════════╗")
    print(f"║ ⚖️  COMPARATOR AGENT & MCP CLIENT                             ║")
    print(f"╠══════════════════════════════════════════════════════════════╣")
    
    import asyncio
    
    # 1. Compare Dataset
    print(f"║ 🛠️  MCP Client: Calling 'compare_dataset'...")
    comparison_str = asyncio.run(run_mcp_tool("compare_dataset", {
        "synthetic_reviews": state["all_synthetic_reviews"],
        "real_reviews": state["real_reviews"]
    }))
    comparison = eval(comparison_str) 
    print(f"║    Verdict: {comparison.get('verdict', 'Analysis Complete')}")
    print(f"║    Similarity: {comparison.get('similarity_to_real', {}).get('avg_max_similarity', 0)}")
    
    # 2. Evaluate Batch
    print(f"║ 🛠️  MCP Client: Calling 'evaluate_reviews'...")
    report_str = asyncio.run(run_mcp_tool("evaluate_reviews", {
        "reviews": state["all_synthetic_reviews"]
    }))
    full_report = eval(report_str)
    
    # 3. Save Report
    print(f"║ 🛠️  MCP Client: Calling 'save_report'...")
    save_msg = asyncio.run(run_mcp_tool("save_report", {
        "report_data": full_report,
        "output_path": "data/quality_report.json"
    }))
    print(f"║    {save_msg}")
    
    print(f"╚══════════════════════════════════════════════════════════════╝")
    
    return {
        **state,
        "comparison_report": comparison
    }


# ----- 4. Edges -----

def check_quality(state: ReviewState) -> str:
    if state.get("quality_assessment", {}).get("pass", False):
        return "accept"
    # If failed, go back to PLANNER (review_planner) if attempts remain
    return "retry_plan" if state["attempt"] < state["max_attempts"] else "accept_poor"

def check_batch(state: ReviewState) -> str:
    return "continue" if state["current_index"] < len(state["plan"]) else "finish"


# ----- 5. Build Graph -----

builder = StateGraph(ReviewState)

builder.add_node("planner", planner_node)
builder.add_node("review_planner", review_planner_node)
builder.add_node("generator", generator_node)
builder.add_node("reviewer", reviewer_node)
builder.add_node("finalize", finalize_node)
builder.add_node("comparator", comparator_node)

builder.set_entry_point("planner")

builder.add_edge("planner", "review_planner")
builder.add_edge("review_planner", "generator")
builder.add_edge("generator", "reviewer")

builder.add_conditional_edges(
    "reviewer",
    check_quality,
    {
        "accept": "finalize", 
        "accept_poor": "finalize", 
        "retry_plan": "review_planner"  # Back to planner on failure!
    }
)

builder.add_conditional_edges(
    "finalize",
    check_batch,
    {"continue": "review_planner", "finish": "comparator"}
)

builder.add_edge("comparator", END)

graph = builder.compile()


def run_agentic_workflow(num_reviews: int, real_reviews: List[str] = None):
    initial_state = {
        "total_reviews": num_reviews,
        "plan": [],
        "current_index": 0,
        "persona": None,
        "rating": None,
        "review": None,
        "quality_assessment": None,
        "attempt": 0,
        "max_attempts": 3,
        "detailed_plan": None,
        "all_synthetic_reviews": [],
        "real_reviews": real_reviews or []
    }
    return graph.invoke(initial_state)
