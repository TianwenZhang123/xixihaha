"""
Wan 2.7 API Client via DashScope (阿里云百炼).

This module provides a video generation backend that calls Wan 2.7 via
DashScope API instead of running a local model. Useful for:
- Higher quality video generation (Wan 2.7 >> Wan 2.1-1.3B)
- No local GPU requirement for video generation
- Quick experiments without downloading model weights

The API is asynchronous: submit task → poll status → download result.

Reference: https://help.aliyun.com/zh/model-studio/legacy-wan-text-to-video-api-reference
"""

import os
import time
import json
import urllib.request
import urllib.error
from typing import Optional, Dict, Any
from pathlib import Path


class WanAPIClient:
    """
    Client for calling Wan 2.7 text-to-video API via DashScope.
    
    Uses DashScope's async task API:
    1. POST to create a video generation task
    2. GET to poll task status
    3. Download the resulting video URL
    
    Supports both DashScope Python SDK and raw HTTP fallback.
    """
    
    # Available Wan models on DashScope
    MODELS = {
        "wan2.7-t2v": "wan2.7-t2v",          # Latest, best quality
        "wan2.6-t2v": "wan2.6-t2v",          # Previous gen
        "wan2.5-t2v": "wan2.5-t2v-pro",      # Older
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "wan2.7-t2v",
        size: str = "1280*720",
        duration: int = 5,
        poll_interval: int = 10,
        max_wait: int = 600,
    ):
        """
        Initialize the Wan API client.
        
        Args:
            api_key: DashScope API key. Falls back to DASHSCOPE_API_KEY env var.
            model: Model name (wan2.7-t2v, wan2.6-t2v, etc.)
            size: Video resolution ("1280*720", "960*480", "1920*1080").
            duration: Video duration in seconds (5, 10, or 15).
            poll_interval: Seconds between status polls.
            max_wait: Maximum wait time in seconds before timeout.
        """
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "DashScope API key required. Pass api_key or set DASHSCOPE_API_KEY env var.\n"
                "Get your key at: https://bailian.console.aliyun.com/#/api-key"
            )
        
        self.model = model
        self.size = size
        self.duration = duration
        self.poll_interval = poll_interval
        self.max_wait = max_wait
        
        # DashScope API endpoints (Beijing region)
        self.base_url = "https://dashscope.aliyuncs.com/api/v1"
        self.submit_url = f"{self.base_url}/services/aigc/video-generation/generation"
        self.task_url = f"{self.base_url}/tasks"
        
        self._try_sdk = True  # Try SDK first, fall back to HTTP
    
    def generate_video(
        self,
        prompt: str,
        output_path: str,
        negative_prompt: Optional[str] = None,
        prompt_extend: bool = True,
        seed: Optional[int] = None,
        shot_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a video from text prompt via Wan 2.7 API.
        
        Args:
            prompt: Text description of the video to generate.
            output_path: Where to save the downloaded video.
            negative_prompt: What to avoid in generation.
            prompt_extend: Enable smart prompt rewriting (recommended).
            seed: Random seed for reproducibility.
            shot_type: "single" or "multi" for multi-shot generation.
            
        Returns:
            Dict with keys:
                - video_path: Path to the saved video file
                - video_url: Original URL from API
                - task_id: DashScope task ID
                - elapsed_time: Time taken in seconds
                - usage: Token/resource usage info
        """
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Try DashScope SDK first
        if self._try_sdk:
            try:
                return self._generate_via_sdk(
                    prompt=prompt,
                    output_path=output_path,
                    negative_prompt=negative_prompt,
                    prompt_extend=prompt_extend,
                    seed=seed,
                    shot_type=shot_type,
                )
            except ImportError:
                print("DashScope SDK not found, using HTTP API...")
                self._try_sdk = False
        
        # Fallback to HTTP
        return self._generate_via_http(
            prompt=prompt,
            output_path=output_path,
            negative_prompt=negative_prompt,
            prompt_extend=prompt_extend,
            seed=seed,
            shot_type=shot_type,
        )
    
    def _generate_via_sdk(
        self,
        prompt: str,
        output_path: str,
        negative_prompt: Optional[str] = None,
        prompt_extend: bool = True,
        seed: Optional[int] = None,
        shot_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate video using DashScope Python SDK."""
        import dashscope
        from dashscope import VideoSynthesis
        
        dashscope.api_key = self.api_key
        
        start_time = time.time()
        print(f"  [Wan API] Submitting task: model={self.model}, size={self.size}, duration={self.duration}s")
        print(f"  [Wan API] Prompt: {prompt[:80]}...")
        
        # Build extra_input parameters
        extra_input = {
            "size": self.size,
            "duration": self.duration,
            "prompt_extend": prompt_extend,
        }
        if shot_type:
            extra_input["shot_type"] = shot_type
        
        # Submit async task
        response = VideoSynthesis.async_call(
            model=self.model,
            prompt=prompt,
            negative_prompt=negative_prompt,
            extra_input=extra_input,
            seed=seed or 42,
        )
        
        if response.status_code != 200:
            raise RuntimeError(
                f"Wan API submit failed: {response.code} - {response.message}"
            )
        
        task_id = response.output.task_id
        print(f"  [Wan API] Task submitted: {task_id}")
        
        # Poll for completion
        elapsed = 0
        while elapsed < self.max_wait:
            time.sleep(self.poll_interval)
            elapsed = time.time() - start_time
            
            result = VideoSynthesis.fetch(task_id)
            status = result.output.task_status
            
            if status == "SUCCEEDED":
                video_url = result.output.video_url
                print(f"  [Wan API] Video ready! ({elapsed:.0f}s elapsed)")
                
                # Download video
                self._download_file(video_url, output_path)
                print(f"  [Wan API] Saved to: {output_path}")
                
                return {
                    "video_path": output_path,
                    "video_url": video_url,
                    "task_id": task_id,
                    "elapsed_time": elapsed,
                    "usage": getattr(result, "usage", {}),
                }
            elif status == "FAILED":
                error_msg = getattr(result.output, "message", "Unknown error")
                raise RuntimeError(f"Wan API generation failed: {error_msg}")
            else:
                print(f"  [Wan API] Status: {status} ({elapsed:.0f}s elapsed)...")
        
        raise TimeoutError(f"Wan API timeout after {self.max_wait}s. Task: {task_id}")
    
    def _generate_via_http(
        self,
        prompt: str,
        output_path: str,
        negative_prompt: Optional[str] = None,
        prompt_extend: bool = True,
        seed: Optional[int] = None,
        shot_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate video using raw HTTP API (no SDK dependency)."""
        start_time = time.time()
        print(f"  [Wan API/HTTP] Submitting task: model={self.model}")
        print(f"  [Wan API/HTTP] Prompt: {prompt[:80]}...")
        
        # Build request body
        input_params = {"prompt": prompt}
        if negative_prompt:
            input_params["negative_prompt"] = negative_prompt
        
        parameters = {
            "size": self.size,
            "duration": self.duration,
            "prompt_extend": prompt_extend,
            "seed": seed or 42,
        }
        if shot_type:
            parameters["shot_type"] = shot_type
        
        body = {
            "model": self.model,
            "input": input_params,
            "parameters": parameters,
        }
        
        # Submit task
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-DashScope-Async": "enable",
        }
        
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(self.submit_url, data=data, headers=headers)
        
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.readable() else ""
            raise RuntimeError(f"Wan API submit failed ({e.code}): {error_body}")
        
        task_id = result.get("output", {}).get("task_id")
        if not task_id:
            raise RuntimeError(f"No task_id in response: {result}")
        
        print(f"  [Wan API/HTTP] Task submitted: {task_id}")
        
        # Poll for completion
        poll_url = f"{self.task_url}/{task_id}"
        poll_headers = {"Authorization": f"Bearer {self.api_key}"}
        
        elapsed = 0
        while elapsed < self.max_wait:
            time.sleep(self.poll_interval)
            elapsed = time.time() - start_time
            
            poll_req = urllib.request.Request(poll_url, headers=poll_headers)
            with urllib.request.urlopen(poll_req, timeout=30) as resp:
                status_result = json.loads(resp.read().decode("utf-8"))
            
            status = status_result.get("output", {}).get("task_status")
            
            if status == "SUCCEEDED":
                video_url = status_result["output"]["video_url"]
                print(f"  [Wan API/HTTP] Video ready! ({elapsed:.0f}s elapsed)")
                
                # Download
                self._download_file(video_url, output_path)
                print(f"  [Wan API/HTTP] Saved to: {output_path}")
                
                return {
                    "video_path": output_path,
                    "video_url": video_url,
                    "task_id": task_id,
                    "elapsed_time": elapsed,
                    "usage": status_result.get("usage", {}),
                }
            elif status == "FAILED":
                error_msg = status_result.get("output", {}).get("message", "Unknown")
                raise RuntimeError(f"Wan API generation failed: {error_msg}")
            else:
                print(f"  [Wan API/HTTP] Status: {status} ({elapsed:.0f}s)...")
        
        raise TimeoutError(f"Wan API timeout after {self.max_wait}s. Task: {task_id}")
    
    def _download_file(self, url: str, save_path: str):
        """Download a file from URL to local path."""
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=120) as resp:
            with open(save_path, "wb") as f:
                while True:
                    chunk = resp.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
    
    def test_connection(self) -> bool:
        """Test if the API key is valid by listing models."""
        try:
            url = f"{self.base_url}/models"
            headers = {"Authorization": f"Bearer {self.api_key}"}
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status == 200
        except Exception as e:
            print(f"  Connection test failed: {e}")
            return False


class MockWanAPIClient:
    """Mock client for testing without API access."""
    
    def __init__(self, **kwargs):
        self.model = kwargs.get("model", "wan2.7-t2v")
        self.call_count = 0
    
    def generate_video(
        self,
        prompt: str,
        output_path: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate a dummy video file for testing."""
        self.call_count += 1
        
        # Create a minimal valid mp4 placeholder
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Write a small dummy file
        with open(output_path, "wb") as f:
            f.write(b"\x00" * 1024)  # Placeholder
        
        print(f"  [Mock Wan API] Generated dummy video #{self.call_count}: {output_path}")
        
        return {
            "video_path": output_path,
            "video_url": "https://mock.example.com/video.mp4",
            "task_id": f"mock-task-{self.call_count}",
            "elapsed_time": 0.1,
            "usage": {"video_duration": 5},
        }
    
    def test_connection(self) -> bool:
        return True
