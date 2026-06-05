import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gemma4_mlx_runtime.llama_cpp_backend import LlamaCppBackend

def test_llama_cpp_light():
    print("test_llama_cpp_light...")
    
    try:
        import llama_cpp
    except ImportError:
        print("  SKIP: llama-cpp-python is not installed")
        sys.exit(0)
        
    print("  llama-cpp-python is installed")
    
    # We can instantiate it without a real model just to check capabilities,
    # but the constructor calls load() which will fail if the model file doesn't exist.
    # Let's mock load to avoid failure
    original_load = LlamaCppBackend.load
    LlamaCppBackend.load = lambda self, **kwargs: None
    
    try:
        backend = LlamaCppBackend(model_path="dummy.gguf")
        
        cap = backend.capabilities()
        assert cap.supports_template_verify is False
        assert cap.supports_snapshot_restore is False
        assert cap.name == "llama_cpp"
        
        stats = backend.stats()
        # Ensure stats is serializable
        json.dumps(stats)
        print("  OK")
    finally:
        LlamaCppBackend.load = original_load

if __name__ == "__main__":
    test_llama_cpp_light()
    print("All tests passed.")
