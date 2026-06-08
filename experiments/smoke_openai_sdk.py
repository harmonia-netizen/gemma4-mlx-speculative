import os
import sys
import time
import urllib.request
import urllib.error
import subprocess
import signal

try:
    import openai
    from openai import OpenAI
except ImportError:
    openai = None

def main():
    if openai is None:
        print("SKIP: openai package is not installed")
        sys.exit(0)
        
    model = os.environ.get("LSR_MODEL")
    if not model:
        print("SKIP: LSR_MODEL is not set")
        sys.exit(0)
        
    backend = os.environ.get("LSR_BACKEND", "mlx")
    if backend in ["gguf", "llama_cpp"]:
        model_type = os.environ.get("LSR_MODEL_TYPE")
        candidate_json = os.environ.get("LSR_CANDIDATE_JSON")
        if not model_type and not candidate_json:
            print("ERROR: LSR_MODEL_TYPE or LSR_CANDIDATE_JSON is required for GGUF")
            sys.exit(1)

    port = 8124
    env = os.environ.copy()
    env["PORT"] = str(port)
    env["PYTHONPATH"] = "."
    
    cmd = [sys.executable, "-m", "local_speculative_runtime.openai_api"]
    
    print(f"Starting server: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr, env=env)
    
    base_url = f"http://127.0.0.1:{port}/v1"
    models_url = f"http://127.0.0.1:{port}/v1/models"
    
    try:
        # Wait for server readiness
        ready = False
        for _ in range(30):
            try:
                req = urllib.request.Request(models_url, method="GET")
                with urllib.request.urlopen(req, timeout=1) as response:
                    if response.status == 200:
                        ready = True
                        break
            except urllib.error.URLError:
                pass
            time.sleep(1)
            
        if not ready:
            print(f"ERROR: Server did not become ready in time on port {port}")
            sys.exit(1)
            
        print("Server is ready. Running checks via OpenAI SDK...")
        
        client = OpenAI(base_url=base_url, api_key="local")
        
        # Check models.list()
        print("Calling client.models.list()...")
        models_response = client.models.list()
        
        model_found = False
        for m in models_response.data:
            if m.id == model or m.id == "local-model":
                model_found = True
                break
                
        if not model_found:
            print("ERROR: Model not found in client.models.list() data.")
            sys.exit(1)
            
        print("OK: client.models.list()")
        
        # Check chat.completions.create()
        print("Calling client.chat.completions.create()...")
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Return exactly: OK"}],
            max_tokens=16,
            stream=False
        )
        
        content = completion.choices[0].message.content
        if content is None:
            print("ERROR: content is None in chat completions response.")
            sys.exit(1)
            
        print(f"OK: client.chat.completions.create() (Response: {content.strip()})")
            
        print("SDK Smoke test completed successfully.")
            
    except Exception as e:
        print(f"Exception during SDK smoke test: {e}")
        sys.exit(1)
    finally:
        # Kill the server
        print("Stopping server...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.kill(process.pid, signal.SIGKILL)
            process.wait()

if __name__ == "__main__":
    main()
