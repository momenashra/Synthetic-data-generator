from dotenv import load_dotenv
import os
import sys
load_dotenv()
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional, Dict, List, Any
from agents import PlannerAgent, GeneratorAgent, ReviewerAgent, ComparatorAgent
import json
from datetime import datetime
import time
import yaml
import numpy as np
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client



# ----- 1. State Definition -----
class ReviewState(TypedDict):
    total_reviews: int
    plan: List[Dict[str, Any]]
    current_index: int
    
    detailed_plan: Optional[Dict[str, Any]]
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

# Global placeholders for agents
planner = None
generator_agent = None
reviewer_agent = None
comparator_agent = None

def get_agents():
    global planner, generator_agent, reviewer_agent, comparator_agent
    if planner is None:
        config = load_config()
        provider1 = os.getenv('LLM_PROVIDER1', 'google').lower()
        provider2 = os.getenv('LLM_PROVIDER2', 'groq').lower()

        planner = PlannerAgent(config=config)
        generator_agent = GeneratorAgent(config=config, provider=provider1)
        reviewer_agent = ReviewerAgent(config=config, provider=provider1)
        comparator_agent = ComparatorAgent(config=config)
    return planner, generator_agent, reviewer_agent, comparator_agent


# ----- 3. Nodes -----

def planner_node(state: ReviewState) -> ReviewState:
    planner, _, _, _ = get_agents()
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
    planner, _, _, _ = get_agents()
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
    _, generator_agent, _, _ = get_agents()
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
    _, _, reviewer_agent, _ = get_agents()
    print(f"\n┌──────────────────────────────────────────────────────────────┐")
    print(f"│ 🔍 REVIEWER AGENT                                            │")
    print(f"├──────────────────────────────────────────────────────────────┤")
    
    assessment = reviewer_agent.review_review(
        review_text=state["review"],
        rating=state["detailed_plan"]["target_rating"],
        plan=state["detailed_plan"]
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
    from models.review_storage import get_review_storage
    
    _, generator_agent, _, _ = get_agents()
    passed = state["quality_assessment"].get("pass", False)
    action = "✅ Accepted" if passed else "⚠️ Accepted (Poor Quality)"
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
    
    # Save immediately with embedding when quality passes
    if passed:
        storage = get_review_storage()
        storage.save_review(review_data, compute_embedding=True)
        print(f"[System]: 💾 Review saved with embedding to storage")
    
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
    _, _, _, comparator_agent = get_agents()
    print(f"\n╔══════════════════════════════════════════════════════════════╗")
    print(f"║ ⚖️  COMPARATOR AGENT & MCP CLIENT                             ║")
    print(f"╠══════════════════════════════════════════════════════════════╣")
    
    # 0. Post-generation Semantic Filtering (Deduplication)
    from quality.diversity import DiversityAnalyzer
    da = DiversityAnalyzer()
    all_texts = [r['text'] for r in state["all_synthetic_reviews"]]
    
    print(f"║ 🔎 Semantic Filtering: Analyzing {len(all_texts)} reviews for redundancy...")
    keep_indices = da.semantic_deduplication(all_texts, threshold=0.92) # Slightly strict threshold
    
    filtered_reviews = [state["all_synthetic_reviews"][i] for i in keep_indices]
    removed_count = len(all_texts) - len(keep_indices)
    
    if removed_count > 0:
        print(f"║ ✨ Deduplication: Removed {removed_count} near-duplicate reviews.")
    else:
        print(f"║ ✅ Deduplication: No redundant reviews found.")

    import asyncio
    
    # 1. Compare Dataset
    print(f"║ 🛠️  MCP Client: Calling 'compare_dataset'...")
    comparison_str = asyncio.run(run_mcp_tool("compare_dataset", {
        "synthetic_reviews": filtered_reviews,
        "real_reviews": state["real_reviews"]
    }))
    comparison = eval(comparison_str) 
    print(f"║    Verdict: {comparison.get('verdict', 'Analysis Complete')}")
    print(f"║    Similarity: {comparison.get('similarity_to_real', {}).get('avg_max_similarity', 0)}")
    
    # 2. Evaluate Batch
    print(f"║ 🛠️  MCP Client: Calling 'evaluate_reviews'...")
    report_str = asyncio.run(run_mcp_tool("evaluate_reviews", {
        "reviews": filtered_reviews
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
    # Ensure agents are instantiated
    get_agents()
    
    # Start timing
    start_time = time.time()

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
    
    result = graph.invoke(initial_state)
    
    # Calculate timing metrics
    end_time = time.time()
    total_time = end_time - start_time
    num_generated = len(result.get('all_synthetic_reviews', []))
    avg_time = total_time / num_generated if num_generated > 0 else 0
    
    # Add performance metrics to result
    result['model_performance'] = {
        'total_time_seconds': round(total_time, 2),
        'avg_time_per_review': round(avg_time, 2),
        'reviews_generated': num_generated,
        'model_name': os.getenv('MODEL_NAME', 'unknown')
    }
    
    return result
