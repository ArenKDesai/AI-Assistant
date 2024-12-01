import ray
import llm_pb2, llm_pb2_grpc, grpc
from datetime import datetime
import time
from concurrent import futures
import logging
from typing import Optional, Dict, Any, List, Tuple
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from dataclasses import dataclass
import numpy as np
from queue import PriorityQueue
from threading import Lock

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Ray
ray.init(
    num_gpus=1,
    runtime_env={
        "pip": ["transformers", "torch"],
    }
)

@dataclass
class BatchConfig:
    max_batch_size: int = 32  # Maximum number of requests in a batch
    max_sequence_length: int = 512  # Maximum sequence length
    max_wait_time: float = 0.1  # Maximum time to wait for batch formation (seconds)
    priority_levels: int = 3  # Number of priority levels
    max_tokens_per_batch: int = 8192  # Maximum total tokens in a batch


# ---------------------------------------------------------------------------
# LLM Server
# ---------------------------------------------------------------------------

class RequestPrioritizer:
    """Manages request priorities based on user quotas and waiting time"""
    def __init__(self, priority_levels: int = 3):
        self.user_quotas: Dict[str, int] = {}  # Requests per minute per user
        self.user_request_times: Dict[str, List[float]] = {}
        self.priority_levels = priority_levels
        self.lock = Lock()
    
    def get_priority(self, user_id: str, wait_time: float) -> int:
        with self.lock:
            current_time = time.time()
            
            # Clean up old request times
            if user_id in self.user_request_times:
                self.user_request_times[user_id] = [
                    t for t in self.user_request_times[user_id]
                    if current_time - t < 60  # Keep last minute
                ]
            
            # Calculate request rate
            request_count = len(self.user_request_times.get(user_id, []))
            quota = self.user_quotas.get(user_id, 30)  # Default 30 requests per minute
            
            # Calculate priority based on usage and wait time
            if request_count >= quota:
                base_priority = self.priority_levels - 1  # Lowest priority
            else:
                base_priority = 0  # Highest priority
            
            # Adjust priority based on wait time
            wait_priority = min(int(wait_time / 5), self.priority_levels - 1)
            
            return max(0, base_priority - wait_priority)

    def record_request(self, user_id: str):
        with self.lock:
            if user_id not in self.user_request_times:
                self.user_request_times[user_id] = []
            self.user_request_times[user_id].append(time.time())

@ray.remote(num_gpus=1)
class GPUWorker:
    """Single GPU worker that handles all LLM operations"""
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-chat-hf"):
        self.device = "cuda"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.batch_config = BatchConfig()
    
    def process_batch(self, 
                     batch_data: List[Tuple[str, str, float]], 
                     max_length: int = 512) -> List[str]:
        """Process a batch of requests"""
        try:
            prompts = [data[1] for data in batch_data]
            
            # Tokenize all prompts
            inputs = self.tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate responses
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_new_tokens=max_length,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.eos_token_id,
                    attention_mask=inputs["attention_mask"]
                )
            
            # Decode responses
            responses = []
            for output in outputs:
                response = self.tokenizer.decode(output, skip_special_tokens=True)
                response = response.split("Assistant:")[-1].strip()
                responses.append(response)
            
            return responses
            
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            return ["Error generating response"] * len(batch_data)

@ray.remote
class BatchScheduler:
    """Manages request batching and scheduling"""
    def __init__(self):
        self.pending_requests: List[Tuple[int, str, str, float]] = []  # (priority, user_id, prompt, timestamp)
        self.batch_config = BatchConfig()
        self.prioritizer = RequestPrioritizer()
    
    def add_request(self, user_id: str, prompt: str) -> int:
        """Add request to pending queue"""
        timestamp = time.time()
        priority = self.prioritizer.get_priority(user_id, 0)
        self.pending_requests.append((priority, user_id, prompt, timestamp))
        self.prioritizer.record_request(user_id)
        return len(self.pending_requests) - 1
    
    def get_next_batch(self) -> List[Tuple[str, str, float]]:
        """Get next batch of requests based on priority and waiting time"""
        if not self.pending_requests:
            return []
        
        current_time = time.time()
        batch = []
        total_tokens = 0
        
        # Sort requests by priority and waiting time
        self.pending_requests.sort(key=lambda x: (
            x[0],  # priority
            current_time - x[3]  # waiting time
        ))
        
        # Form batch
        remaining_requests = []
        for req in self.pending_requests:
            priority, user_id, prompt, timestamp = req
            
            # Estimate tokens
            estimated_tokens = len(prompt.split()) * 1.5  # Rough estimation
            
            if (len(batch) < self.batch_config.max_batch_size and 
                total_tokens + estimated_tokens <= self.batch_config.max_tokens_per_batch):
                batch.append((user_id, prompt, timestamp))
                total_tokens += estimated_tokens
            else:
                remaining_requests.append(req)
        
        self.pending_requests = remaining_requests
        return batch

class RayLLMProcessor:
    """Manages LLM processing using Ray with single GPU"""
    def __init__(self):
        self.gpu_worker = GPUWorker.remote()
        self.batch_scheduler = BatchScheduler.remote()
        self.conversation_cache: Dict[str, list] = {}
        
        # Start batch processing thread
        self.processing_thread = threading.Thread(target=self._process_batches, daemon=True)
        self.processing_thread.start()
    
    def _process_batches(self):
        """Background thread for batch processing"""
        while True:
            try:
                # Get next batch
                batch = ray.get(self.batch_scheduler.get_next_batch.remote())
                if not batch:
                    time.sleep(0.01)
                    continue
                
                # Process batch
                responses = ray.get(self.gpu_worker.process_batch.remote(batch))
                
                # Update conversation cache
                for (user_id, prompt, _), response in zip(batch, responses):
                    if user_id in self.conversation_cache:
                        self.conversation_cache[user_id].extend([
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": response}
                        ])
                
            except Exception as e:
                logger.error(f"Error in batch processing: {e}")
                time.sleep(0.1)
    
    def process_query(self, user_id: str, query: str) -> str:
        """Process a single query"""
        try:
            # Initialize conversation history if needed
            if user_id not in self.conversation_cache:
                self.conversation_cache[user_id] = []
            
            # Construct prompt with history
            prompt = self._construct_prompt(self.conversation_cache[user_id], query)
            
            # Add to batch scheduler
            request_id = ray.get(self.batch_scheduler.add_request.remote(user_id, prompt))
            
            # Wait for response
            max_wait = 30  # seconds
            start_time = time.time()
            
            while time.time() - start_time < max_wait:
                if user_id in self.conversation_cache:
                    history = self.conversation_cache[user_id]
                    if len(history) > 0 and history[-1]["content"] != query:
                        return history[-1]["content"]
                time.sleep(0.05)
            
            raise TimeoutError("Request processing timed out")
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"Error processing query: {str(e)}"
    
    def _construct_prompt(self, conversation: list, query: str) -> str:
        """Construct prompt with conversation history"""
        return f"User: {query}\nAssistant:"

class LLM(llm_pb2_grpc.LLMServicer):
    """
    Main class for communication with the LLM.

    process_query is necessary for basic call/response, 
    but we could add more communication protocols for 
    stuff like uploading files. 
    """
    def __init__(self):
        self.processor = RayLLMProcessor()
    
    def process_query(self, request, context):
        try:
            query = request.request
            user_id = getattr(request, 'user_id', None) or 'default'
            
            response = self.processor.process_query(user_id, query)
            
            return llm_pb2.QueryResp(response=response)
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return llm_pb2.QueryResp(response="Error processing request")

def serve(port: str = "0.0.0.0:5440", max_workers: int = 10):
    """Start the gRPC server"""
    try:
        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=max_workers),
            options=[
                ("grpc.so_reuseport", 0),
                ('grpc.max_send_message_length', 50 * 1024 * 1024),
                ('grpc.max_receive_message_length', 50 * 1024 * 1024)
            ]
        )
        
        llm_pb2_grpc.add_LLMServicer_to_server(LLM(), server)
        server.add_insecure_port(port)
        
        logger.info(f"Starting server on {port} at {datetime.now().ctime()}")
        server.start()
        
        def shutdown(signo, frame):
            logger.info("Shutting down server...")
            ray.shutdown()
            server.stop(0)
        
        import signal
        signal.signal(signal.SIGINT, shutdown)
        signal.signal(signal.SIGTERM, shutdown)
        
        server.wait_for_termination()
        
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        ray.shutdown()
        raise

if __name__ == '__main__':
    print(f"Starting server on {datetime.now().ctime()}")

    PORT = os.getenv("LLM_SERVER_PORT", "0.0.0.0:5440")
    MAX_WORKERS = int(os.getenv("LLM_SERVER_WORKERS", "10")) 

    serve(PORT, MAX_WORKERS)   