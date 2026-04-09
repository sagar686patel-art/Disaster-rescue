"""
Mandatory Inference Script
Implements required [START], [STEP], [END] logging protocol.
Uses OpenAI Client for LLM-based action selection.
Supports all three difficulty levels: easy, medium, hard.
"""

import os
import sys
import uuid
import json
from typing import Dict, Any, Optional, Tuple
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment.disaster_env import DisasterRescueEnv
from utils.logger import StructuredLogger, EpisodeLogger
from utils.graders import get_grader
from configs.task_config import get_all_difficulties
from agents.baseline_agent import get_agent


class InferenceRunner:
    """
    Main inference runner implementing mandatory logging protocol.
    
    Mandatory Log Format:
    [START] run_id=<uuid> task=<task_id> model=<MODEL_NAME>
    [STEP] step=<int> reward=<float> state=<json> action=<int>
    [END] run_id=<uuid> task=<task_id> score=<float>
    """
    
    def __init__(
        self,
        api_base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        hf_token: Optional[str] = None,
        use_llm: bool = True,
    ):
        """
        Initialize inference runner.
        
        Args:
            api_base_url: OpenAI API base URL (from env)
            model_name: Model name (from env)
            hf_token: Hugging Face token (from env)
            use_llm: If True, use LLM for actions; else use baseline agent (default: True)
        """
        # Use environment variables provided by validator (REQUIRED)
        self.api_base_url = os.getenv("API_BASE_URL")
        self.api_key = os.getenv("API_KEY")
        self.model_name = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
        self.hf_token = hf_token or os.getenv("HF_TOKEN", "")
        
        # Initialize logger (stdout only, per requirements)
        self.logger = StructuredLogger(use_stdout=True)
        
        print(f"[DEBUG] API_BASE_URL: {self.api_base_url}", file=sys.stderr)
        print(f"[DEBUG] API_KEY exists: {bool(self.api_key)}", file=sys.stderr)
        print(f"[DEBUG] MODEL_NAME: {self.model_name}", file=sys.stderr)
        
        # Validate that validator has provided credentials
        if not self.api_base_url or not self.api_key:
            print("[ERROR] API_BASE_URL or API_KEY not provided by validator", file=sys.stderr)
            self.use_llm = False
            self.client = None
            return
        
        # Initialize OpenAI client with validator-provided credentials
        self.use_llm = use_llm
        self.client = None
        
        if self.use_llm:
            try:
                from openai import OpenAI
                # Initialize with VALIDATOR'S API_BASE_URL and API_KEY
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.api_base_url,
                )
                print(f"[INFO] LLM client initialized successfully", file=sys.stderr)
                print(f"[INFO] Using base_url: {self.api_base_url}", file=sys.stderr)
                print(f"[INFO] Using model: {self.model_name}", file=sys.stderr)
            except Exception as e:
                print(f"[ERROR] Failed to initialize LLM client: {e}", file=sys.stderr)
                self.use_llm = False
                self.client = None
        
        self.run_id = str(uuid.uuid4())
        self.results = {}
    
    def run_episode(
        self,
        difficulty: str,
        seed: Optional[int] = None,
        max_steps: int = 500,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Run a single episode and return score.
        
        Args:
            difficulty: "easy", "medium", or "hard"
            seed: Random seed for reproducibility
            max_steps: Maximum steps per episode
        
        Returns:
            Tuple of (score, info_dict)
        """
        # Create environment
        env = DisasterRescueEnv(difficulty=difficulty, seed=seed)
        
        # Create episode logger
        episode_logger = EpisodeLogger(
            self.logger,
            run_id=self.run_id,
            task=difficulty,
            model=self.model_name,
        )
        
        # Log episode start
        episode_logger.start()
        
        # Reset environment
        obs, env_info = env.reset(seed=seed)
        
        # Create agent (baseline for comparison)
        agent = get_agent("greedy", env=env, seed=seed)
        
        total_reward = 0.0
        step = 0
        
        # Episode loop
        while step < max_steps:
            # ALWAYS try to use LLM first if available
            if self.use_llm and self.client:
                action = self._get_llm_action(obs, difficulty, step, env_info)
            else:
                # Only fallback to baseline if LLM not available
                action, _ = agent.predict(obs)
            
            # Step environment
            obs, reward, terminated, truncated, env_info = env.step(action)
            total_reward += reward
            
            # Get environment state for logging
            env_state = env.state()
            
            # Log step (mandatory format)
            episode_logger.log_step(
                reward=float(reward),
                state=env_state,
                action=int(action),
            )
            
            step += 1
            
            # Check termination
            if terminated or truncated:
                break
        
        # Grade the episode
        grader = get_grader(difficulty)
        final_score = grader.grade(env.state())
        
        # Log episode end (mandatory format)
        episode_logger.end(final_score=final_score)
        
        info = {
            "difficulty": difficulty,
            "steps": step,
            "total_reward": total_reward,
            "score": final_score,
            "env_state": env.state(),
        }
        
        return final_score, info
    
    def _get_llm_action(
        self,
        obs: np.ndarray,
        difficulty: str,
        step: int,
        env_info: Dict[str, Any],
    ) -> int:
        """
        Get action from LLM using OpenAI client.
        Makes API call to validator's LiteLLM proxy.
        
        Args:
            obs: Current observation
            difficulty: Current task difficulty
            step: Current step number
            env_info: Environment info
        
        Returns:
            Action index (0-7)
        """
        if not self.use_llm or self.client is None:
            # Fallback to random if LLM unavailable
            return np.random.randint(0, 8)
        
        try:
            # Prepare prompt
            prompt = self._prepare_llm_prompt(obs, difficulty, step, env_info)
            
            # Make API call to validator's LiteLLM proxy
            print(f"[DEBUG] Making LLM API call to {self.api_base_url}", file=sys.stderr)
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an AI controlling an autonomous drone in a disaster rescue simulation. "
                                   "Choose the best action given the current state. "
                                   "Respond with a single number 0-7.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                max_tokens=10,
                temperature=0.7,
            )
            
            # Parse response
            action_str = response.choices[0].message.content.strip()
            action = int(action_str)
            print(f"[DEBUG] LLM action: {action}", file=sys.stderr)
            return np.clip(action, 0, 7)
        
        except Exception as e:
            print(f"[ERROR] LLM API call failed: {e}", file=sys.stderr)
            print(f"[ERROR] Exception type: {type(e).__name__}", file=sys.stderr)
            # Fallback to random action
            return np.random.randint(0, 8)
    
    def _prepare_llm_prompt(
        self,
        obs: np.ndarray,
        difficulty: str,
        step: int,
        env_info: Dict[str, Any],
    ) -> str:
        """
        Prepare prompt for LLM.
        
        Args:
            obs: Current observation
            difficulty: Task difficulty
            step: Current step
            env_info: Environment info
        
        Returns:
            Formatted prompt string
        """
        prompt = f"""
Disaster Rescue Simulation - Step {step}
Difficulty: {difficulty}

Current Status:
- Battery: {env_info.get('battery', 'Unknown')}
- Victims Rescued: {env_info.get('victims_rescued', 0)}/{env_info.get('total_victims', 1)}

Available Actions:
0: North, 1: Northeast, 2: East, 3: Southeast
4: South, 5: Southwest, 6: West, 7: Northwest

Observation Summary:
- Victims visible: {np.sum(obs[:,:,1]) > 0}
- Hazards nearby: {np.sum(obs[:,:,2]) > 0}
- Resources available: {np.sum(obs[:,:,3]) > 0}
- Battery low: {env_info.get('battery', 100) < 200}

What action should the drone take? (respond with single number 0-7)
"""
        return prompt
    
    def run_all_tasks(self, seed: Optional[int] = None) -> Dict[str, float]:
        """
        Run all three difficulty levels sequentially.
        
        Args:
            seed: Random seed for reproducibility
        
        Returns:
            Dictionary mapping difficulty -> score
        """
        results = {}
        
        print("[INFO] Starting inference on all tasks with LLM...", file=sys.stderr)
        
        for difficulty in get_all_difficulties():
            print(f"[INFO] Running {difficulty} task...", file=sys.stderr)
            
            try:
                score, info = self.run_episode(
                    difficulty=difficulty,
                    seed=seed,
                )
                results[difficulty] = score
                print(
                    f"[INFO] {difficulty.upper()} task completed. Score: {score:.4f}",
                    file=sys.stderr,
                )
            except Exception as e:
                print(f"[ERROR] {difficulty} task failed: {e}", file=sys.stderr)
                results[difficulty] = 0.0
        
        self.results = results
        return results
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of all runs.
        
        Returns:
            Summary dictionary
        """
        if not self.results:
            return {"error": "No results yet"}
        
        scores = list(self.results.values())
        return {
            "run_id": self.run_id,
            "model": self.model_name,
            "results": self.results,
            "mean_score": float(np.mean(scores)) if scores else 0.0,
            "min_score": float(np.min(scores)) if scores else 0.0,
            "max_score": float(np.max(scores)) if scores else 0.0,
        }


def main():
    """
    Main entry point for inference.
    
    Environment Variables (MUST be provided by validator):
    - API_BASE_URL: LiteLLM proxy base URL (REQUIRED)
    - API_KEY: LiteLLM proxy API key (REQUIRED)
    - MODEL_NAME: Model name (default: gpt-3.5-turbo)
    """
    # Get configuration from environment
    # USE_LLM is ALWAYS True - we must make API calls through validator's proxy
    use_llm = True
    seed = int(os.getenv("SEED", "0")) if os.getenv("SEED") else None
    
    # Create runner with LLM enabled by default
    runner = InferenceRunner(use_llm=use_llm)
    
    # Run all tasks
    results = runner.run_all_tasks(seed=seed)
    
    # Print summary to stderr
    summary = runner.get_summary()
    print("[INFO] Inference Complete!", file=sys.stderr)
    print(json.dumps(summary, indent=2), file=sys.stderr)
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
