import os
import sys
import time
import json
import urllib.request
import urllib.error
import subprocess
import signal

def main():
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

    port = 8123
    env = os.environ.copy()
    env["PORT"] = str(port)
    env["PYTHONPATH"] = "."
    
    cmd = [sys.executable, "-m", "local_speculative_runtime.openai_api"]
    
    print(f"Starting server: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr, env=env)
    
    base_url = f"http://127.0.0.1:{port}"
    models_url = f"{base_url}/v1/models"
    chat_url = f"{base_url}/v1/chat/completions"
    
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
            
        print("Server is ready. Running checks...")
        
        # Check /v1/models
        req = urllib.request.Request(models_url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            if data.get("object") != "list":
                print(f"ERROR: /v1/models object is not 'list'. Got: {data.get('object')}")
                sys.exit(1)
            if not data.get("data") or len(data["data"]) == 0:
                print("ERROR: /v1/models returned empty data list.")
                sys.exit(1)
            print("OK: /v1/models")
            
        # Check /v1/chat/completions (stream: false)
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "Return exactly: OK"}],
            "max_tokens": 16,
            "stream": False
        }
        data_json = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(chat_url, data=data_json, method="POST")
        req.add_header("Content-Type", "application/json")
        
        with urllib.request.urlopen(req, timeout=60) as response:
            res_data = json.loads(response.read().decode())
            choices = res_data.get("choices", [])
            if not choices:
                print("ERROR: No choices returned in chat completions.")
                sys.exit(1)
            content = choices[0].get("message", {}).get("content")
            if content is None:
                print("ERROR: content is None in chat completions response.")
                sys.exit(1)
            
            # Verify warning header is absent
            if response.headers.get("X-LSR-Warning"):
                print(f"ERROR: Found X-LSR-Warning header in stream=False response: {response.headers.get('X-LSR-Warning')}")
                sys.exit(1)
                
            print(f"OK: /v1/chat/completions (stream=False) (Response: {content.strip()})")
            
        # Check /v1/chat/completions (stream: true fallback)
        payload["stream"] = True
        data_json = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(chat_url, data=data_json, method="POST")
        req.add_header("Content-Type", "application/json")
        
        with urllib.request.urlopen(req, timeout=60) as response:
            if response.status != 200:
                print(f"ERROR: stream=true returned status {response.status}")
                sys.exit(1)
                
            warning_header = response.headers.get("X-LSR-Warning")
            if not warning_header:
                print("ERROR: X-LSR-Warning header is missing in stream=True response")
                sys.exit(1)
            if "stream=true is not supported" not in warning_header:
                print(f"ERROR: Unexpected X-LSR-Warning header value: {warning_header}")
                sys.exit(1)
                
            res_data = json.loads(response.read().decode())
            choices = res_data.get("choices", [])
            if not choices:
                print("ERROR: No choices returned in chat completions (stream=True).")
                sys.exit(1)
            content = choices[0].get("message", {}).get("content")
            if content is None:
                print("ERROR: content is None in chat completions response (stream=True).")
                sys.exit(1)
                
            print(f"OK: /v1/chat/completions (stream=True fallback) (Response: {content.strip()})")
            
        print("Smoke test completed successfully.")
            
    except urllib.error.HTTPError as e:
        print(f"HTTPError: {e.code} - {e.read().decode()}")
        sys.exit(1)
    except Exception as e:
        print(f"Exception during smoke test: {e}")
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
