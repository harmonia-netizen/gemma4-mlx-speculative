import os
import sys
import time
import uuid
import logging
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field
import uvicorn

from local_speculative_runtime.session_cache import SessionCacheAPI

logger = logging.getLogger(__name__)

# --- Pydantic Models ---

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 128
    temperature: Optional[float] = 0.0
    stream: Optional[bool] = False
    
    # Custom extensions
    backend: Optional[str] = None
    draft_block_size: Optional[int] = 8
    template_min_tokens: Optional[int] = 1

class ChatCompletionResponseChoiceMessage(BaseModel):
    role: str = "assistant"
    content: str

class ChatCompletionResponseChoice(BaseModel):
    index: int = 0
    message: ChatCompletionResponseChoiceMessage
    finish_reason: str = "stop"

class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: ChatCompletionUsage
    
    # Custom metadata for debugging/inspection
    metadata: Optional[Dict[str, Any]] = None

# --- Helper Functions ---

class ModelObject(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "local-speculative-runtime"

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelObject]


def format_messages_to_prompt(messages: List[ChatMessage]) -> str:
    """
    Very basic message to prompt conversion.
    In a real production environment, this should be replaced with
    a proper chat template renderer from the model's tokenizer.
    """
    prompt = ""
    for msg in messages:
        if msg.role == "system":
            prompt += f"System: {msg.content}\n\n"
        elif msg.role == "user":
            prompt += f"User: {msg.content}\n\n"
        elif msg.role == "assistant":
            prompt += f"Assistant: {msg.content}\n\n"
    prompt += "Assistant:"
    return prompt

def split_messages_for_session(messages: List[ChatMessage]) -> tuple[str, str]:
    """
    Splits the chat messages into a reusable prefix and the latest suffix.
    The prefix contains all messages except the last one.
    The suffix contains the last message, and ends with "Assistant:".
    If there is only 1 message, prefix is empty.
    """
    if not messages:
        raise ValueError("messages list cannot be empty")
        
    prefix_prompt = ""
    suffix_prompt = ""
    
    # Prefix: All messages except the last one
    for msg in messages[:-1]:
        if msg.role == "system":
            prefix_prompt += f"System: {msg.content}\n\n"
        elif msg.role == "user":
            prefix_prompt += f"User: {msg.content}\n\n"
        elif msg.role == "assistant":
            prefix_prompt += f"Assistant: {msg.content}\n\n"
            
    # Suffix: The last message
    last_msg = messages[-1]
    if last_msg.role == "system":
        suffix_prompt += f"System: {last_msg.content}\n\n"
    elif last_msg.role == "user":
        suffix_prompt += f"User: {last_msg.content}\n\n"
    elif last_msg.role == "assistant":
        suffix_prompt += f"Assistant: {last_msg.content}\n\n"
        
    suffix_prompt += "Assistant:"
    
    return prefix_prompt, suffix_prompt


# --- App Factory ---

def create_app(api: Optional[SessionCacheAPI] = None) -> FastAPI:
    app = FastAPI(title="Local Speculative Runtime API", version="0.1.0")
    
    # If API is not provided, initialize it from environment variables
    # This prevents loading the model during test collection or module import
    app.state.api = api
    app.state.api_initialized = False if api is None else True
    app.state.model_id = os.environ.get("LSR_MODEL", "local-model")
    
    def get_api() -> SessionCacheAPI:
        if not app.state.api_initialized:
            backend = os.environ.get("LSR_BACKEND", "mlx")
            model = os.environ.get("LSR_MODEL")
            if not model:
                raise RuntimeError("LSR_MODEL environment variable must be set")
            
            candidate_json = os.environ.get("LSR_CANDIDATE_JSON")
            model_type = os.environ.get("LSR_MODEL_TYPE")
            generation_mode = os.environ.get("LSR_GENERATION_MODE", "low-level")
            
            if generation_mode not in ["low-level", "high-level"]:
                raise RuntimeError(f"LSR_GENERATION_MODE must be 'low-level' or 'high-level', got {generation_mode}")
            
            if backend == "mlx" and generation_mode == "high-level":
                raise RuntimeError("LSR_GENERATION_MODE=high-level cannot be specified with backend='mlx'")
            
            if backend in ["llama_cpp", "gguf"]:
                if model_type == "qwen" and not candidate_json:
                    candidate_json = "experiments/template_candidates_gguf_qwen.json"
                elif not candidate_json:
                    raise RuntimeError("LSR_CANDIDATE_JSON or LSR_MODEL_TYPE=qwen is required for GGUF")
                app.state.api = SessionCacheAPI.load(
                    model_path=model,
                    backend=backend,
                    candidate_json_path=candidate_json,
                    generation_mode=generation_mode
                )
            else:
                app.state.api = SessionCacheAPI.load(model_path=model, backend=backend)
                
            app.state.api_initialized = True
        return app.state.api

    @app.get("/v1/models", response_model=ModelList)
    async def list_models():
        return ModelList(
            data=[
                ModelObject(
                    id=app.state.model_id,
                    created=int(time.time())
                )
            ]
        )

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    async def chat_completions(request: ChatCompletionRequest, response: Response):
        if request.stream:
            response.headers["X-LSR-Warning"] = "stream=true is not supported; returned non-streaming response"
            
        # 1. Split messages into prefix and suffix
        try:
            prefix_text, suffix_text = split_messages_for_session(request.messages)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        # 2. Get API and prepare session
        try:
            api_instance = get_api()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to initialize model: {e}")
            
        session_id = f"chat-{uuid.uuid4()}"
        
        # Create session with the prefix
        create_res = api_instance.create_session(session_id, prefix_text=prefix_text)
        if not create_res.get("ok"):
            api_instance.clear_session(session_id)
            raise HTTPException(status_code=500, detail=f"Failed to create session: {create_res.get('error')}")
            
        # 3. Generate using the suffix
        gen_res = api_instance.generate(
            session_id=session_id,
            suffix_text=suffix_text,
            max_tokens=request.max_tokens,
            draft_block_size=request.draft_block_size,
            template_min_tokens=request.template_min_tokens,
            trace=False
        )
        
        # 4. Cleanup
        api_instance.clear_session(session_id)
        
        if not gen_res.get("ok"):
            raise HTTPException(status_code=500, detail=f"Generation failed: {gen_res.get('error')}")
            
        # 5. Format Response
        completion_text = gen_res.get("text", "")
        prompt_tokens = gen_res.get("prompt_tokens", 0)
        completion_tokens = gen_res.get("completion_tokens", 0)
        total_tokens = prompt_tokens + completion_tokens
        
        response_id = f"chatcmpl-{uuid.uuid4()}"
        
        return ChatCompletionResponse(
            id=response_id,
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatCompletionResponseChoiceMessage(
                        role="assistant",
                        content=completion_text
                    ),
                    finish_reason="stop"
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens
            ),
            metadata=gen_res.get("metadata", {})
        )

    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "127.0.0.1")
    logger.info(f"Starting Local Speculative Runtime API on {host}:{port}")
    
    app = create_app()
    uvicorn.run(app, host=host, port=port)
