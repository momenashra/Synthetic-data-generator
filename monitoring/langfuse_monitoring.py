"""
LangFuse monitoring integration for LLM call tracking and cost calculation.
LangFuse is better suited for custom wrapper classes as it supports explicit usage ingestion.
"""
import os
import sys
import warnings
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Suppress Pydantic v1 warnings if on Python 3.14+
warnings.filterwarnings('ignore', message='Core Pydantic V1 functionality')

load_dotenv()

# Try to import LangFuse
LANGFUSE_AVAILABLE = False
langfuse_client: Optional[Any] = None
Langfuse = None

try:
    from langfuse import Langfuse
    
    # Check version compatibility
    import langfuse as lf_module
    langfuse_version = getattr(lf_module, '__version__', 'unknown')
    print(f"📦 LangFuse version: {langfuse_version}")
    
    LANGFUSE_AVAILABLE = True
    print("✅ LangFuse imported successfully")
    
except ImportError as e:
    print(f"⚠️  LangFuse not installed: {e}")
    print(f"⚠️  Install with: pip install langfuse")
    print(f"⚠️  Monitoring will be disabled.")
except Exception as e:
    print(f"⚠️  LangFuse initialization failed: {e}")
    print(f"⚠️  Monitoring will be disabled.")

def check_pydantic_version():
    """Check Pydantic version for debugging."""
    try:
        import pydantic
        pydantic_version = getattr(pydantic, '__version__', 'unknown')
        print(f"📦 Pydantic version: {pydantic_version}")
        return pydantic_version
    except ImportError:
        print("⚠️  Pydantic not found")
        return None

def init_langfuse_monitoring():
    """Initialize LangFuse monitoring with enhanced error handling."""
    global langfuse_client
    
    if not LANGFUSE_AVAILABLE:
        print("⚠️  LangFuse not available - skipping initialization")
        return None
    
    # Check Pydantic version for debugging
    check_pydantic_version()
    
    langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    langfuse_host = os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL") or "https://cloud.langfuse.com"
    
    if not langfuse_secret_key or not langfuse_public_key:
        print("⚠️  LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY not found in environment")
        print("⚠️  Set these in your .env file to enable monitoring")
        return None
    
    try:
        langfuse_client = Langfuse(
            secret_key=langfuse_secret_key,
            public_key=langfuse_public_key,
            host=langfuse_host
        )
        print(f"🚀 LangFuse monitoring initialized successfully!")
        print(f"   Host: {langfuse_host}")
        print(f"   Public Key: {langfuse_public_key[:10]}...")
        return langfuse_client
        
    except Exception as e:
        print(f"❌ Failed to initialize LangFuse client: {e}")
        print(f"   This may be due to:")
        print(f"   1. Invalid API keys")
        print(f"   2. Network connectivity issues")
        print(f"   3. Incorrect host URL")
        import traceback
        traceback.print_exc()
        return None

def log_llm_call_to_langfuse(
    model: str,
    prompt: str,
    response: str,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int = None,
    provider: str = None,
    metadata: Optional[Dict[str, Any]] = None,
    trace_id: Optional[str] = None,
    generation_id: Optional[str] = None
):
    """
    Log LLM call to LangFuse with explicit token usage and cost tracking.
    
    This is the key advantage of LangFuse - you can explicitly pass usage details
    even with custom wrapper classes, and LangFuse will calculate costs automatically.
    """
    global langfuse_client
    
    if not LANGFUSE_AVAILABLE:
        return None
    
    if not langfuse_client:
        # Try to initialize if not already done
        init_langfuse_monitoring()
        if not langfuse_client:
            return None
    
    try:
        total = total_tokens if total_tokens is not None else (input_tokens + output_tokens)
        
        # LangFuse supports explicit usage_details which is perfect for custom wrappers
        usage_details = {
            "input": input_tokens,
            "output": output_tokens,
            "total": total,
        }
        
        # Create or get trace
        if trace_id:
            trace = langfuse_client.trace(id=trace_id)
        else:
            trace = langfuse_client.trace(
                name="llm_call",
                metadata=metadata or {}
            )
        
        # Create generation with explicit usage details
        generation = trace.generation(
            name=f"{provider or 'unknown'}_{model}",
            model=model,
            input=prompt,
            output=response,
            usage=usage_details,  # Explicit usage ingestion
            metadata={
                "provider": provider or "unknown",
                **(metadata or {})
            }
        )
        
        # Return generation ID for tracking
        gen_id = generation.id if hasattr(generation, 'id') else None
        if gen_id:
            print(f"✅ Logged to LangFuse - Generation ID: {gen_id}")
        return gen_id
        
    except Exception as e:
        print(f"❌ Error logging LLM call to LangFuse: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_langfuse_callback():
    """
    Returns a LangFuse callback handler for LangChain/LangGraph.
    
    Note: In LangFuse 3.x, the callback is imported from langfuse.callback
    """
    if not LANGFUSE_AVAILABLE:
        return None
    
    try:
        from langfuse.callback import CallbackHandler
        
        langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        langfuse_host = os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL") or "https://cloud.langfuse.com"
        
        if langfuse_secret_key and langfuse_public_key:
            return CallbackHandler(
                secret_key=langfuse_secret_key,
                public_key=langfuse_public_key,
                host=langfuse_host
            )
    except Exception as e:
        print(f"⚠️  Failed to create LangFuse callback: {e}")
    
    return None

def flush_langfuse():
    """Flush any pending LangFuse events."""
    global langfuse_client
    if langfuse_client:
        try:
            print("🔄 Flushing LangFuse events...")
            langfuse_client.flush()
            print("✅ LangFuse events flushed successfully")
        except Exception as e:
            print(f"⚠️  Error flushing LangFuse: {e}")

def shutdown_langfuse():
    """Properly shutdown LangFuse client."""
    global langfuse_client
    if langfuse_client:
        try:
            print("🛑 Shutting down LangFuse...")
            langfuse_client.shutdown()
            print("✅ LangFuse shutdown successfully")
        except Exception as e:
            print(f"⚠️  Error shutting down LangFuse: {e}")

# Diagnostic function
def diagnose_langfuse():
    """Run diagnostics to help troubleshoot LangFuse issues."""
    print("\n" + "="*60)
    print("🔍 LangFuse Diagnostics")
    print("="*60)
    
    # Check Python version
    print(f"\n0. Python Version: {sys.version}")
    
    # Check LangFuse availability
    print(f"\n1. LangFuse Available: {LANGFUSE_AVAILABLE}")
    
    if LANGFUSE_AVAILABLE:
        import langfuse as lf_module
        print(f"   Version: {getattr(lf_module, '__version__', 'unknown')}")
    
    # Check Pydantic
    pydantic_version = check_pydantic_version()
    
    # Check environment variables
    print(f"\n2. Environment Variables:")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    
    print(f"   LANGFUSE_SECRET_KEY: {'✅ Set' if secret_key else '❌ Not set'}")
    print(f"   LANGFUSE_PUBLIC_KEY: {'✅ Set' if public_key else '❌ Not set'}")
    print(f"   LANGFUSE_HOST: {host}")
    
    # Check client initialization
    print(f"\n3. Client Initialized: {'✅ Yes' if langfuse_client else '❌ No'}")
    
    # Recommendations
    print(f"\n4. Recommendations:")
    if not LANGFUSE_AVAILABLE:
        print("   ❌ Install LangFuse: pip install langfuse")
    if not secret_key or not public_key:
        print("   ❌ Set API keys in .env file")
    if sys.version_info >= (3, 14):
        print("   ⚠️  Python 3.14+ detected - use Python 3.12 for best compatibility")
    
    print("\n" + "="*60 + "\n")

# Test function
def test_langfuse():
    """Test LangFuse integration with a simple call."""
    print("\n🧪 Testing LangFuse Integration...\n")
    
    # Initialize
    client = init_langfuse_monitoring()
    if not client:
        print("❌ Failed to initialize LangFuse")
        return False
    
    # Test logging
    try:
        result = log_llm_call_to_langfuse(
            model="test-model",
            prompt="Test prompt",
            response="Test response",
            input_tokens=10,
            output_tokens=20,
            provider="test"
        )
        
        if result:
            print(f"✅ Test successful! Generation ID: {result}")
            flush_langfuse()
            return True
        else:
            print("⚠️  Test completed but no generation ID returned")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    diagnose_langfuse()
    test_langfuse()