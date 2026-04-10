"""
Mandatory Inference Script
Implements required [START], [STEP], [END] logging protocol.
FORCES OpenAI LLM API calls through validator's LiteLLM proxy.
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


class InferenceRunner:
    """
    Main inference runner implementing mandatory logging protocol.
    
    Mandatory Log Format:
    [START] run_id=<uuid> task=<task_id> model=<MODEL_NAME>
    [STEP] step=<int> reward=<float> state=<json> action=<int>
    [END] run_id=<uuid> task=<task_id> score=<float>
    
    FORCES API calls through validator's LiteLLM proxy.
    """
    
    def __init__(self):
        """
        Initialize inference runner.
        REQUIRES environment variables: API_BASE_URL, MODEL_NAME, API_KEY
        """
        # Get validator-provided credentials
        self.api_base_url = os.environ.get("API_BASE_URL")
        self.model_name = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")
        self.api_key = os.environ.get("API_KEY")
        
        # Print diagnostics to stderr
        print(f"[INFO] API_BASE_URL from environment: {self.api_base_url}", file=sys.stderr)
        print(f"[INFO] API_KEY from environment: {'SET' if self.api_key else 'NOT SET'}", file=sys.stderr)
        print(f"[INFO] MODEL_NAME: {self.model_name}", file=sys.stderr)
        
        # Verify we have credentials
        if not self.api_base_url:
            raise ValueError("FATAL: API_BASE_URL environment variable not set by validator")
        if not self.api_key:
            raise ValueError("FATAL: API_KEY environment variable not set by validator")
        
        # Initialize logger
        self.logger = StructuredLogger(use_stdout=True)
        
        # Initialize OpenAI client with validator's credentials
        try:
            from openai import OpenAI
            
            # CRITICAL FIX: Only pass api_key and base_url
            # Do NOT pass any other parameters like 'proxies' or 'timeout'
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base_url,
            )
            
            print(f"[INFO] OpenAI client successfully initialized", file=sys.stderr)
            print(f"[INFO] Client will call: {self.api_base_url}", file=sys.stderr)
            
        except ImportError as e:
            raise ImportError(f"OpenAI package not installed: {e}")
        except TypeError as e:
            # This catches the "unexpected keyword argument" errors
            print(f"[ERROR] OpenAI client initialization error: {e}", file=sys.stderr)
            print(f"[ERROR] This may be due to OpenAI version mismatch", file=sys.stderr)
            raise Exception(f"Failed to initialize OpenAI client: {e}")
        except Exception as e:
            raise Exception(f"Failed to initialize OpenAI client: {e}")
        
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
        Makes LLM API calls for every action decision.
        
        Args:
            difficulty: "easy", "medium", or "hard"
            seed: Random seed for reproducibility
            max_steps: Maximum steps per episode
        
        Returns:
            Tuple of (score, info_dict)
        """
        print(f"[INFO] Starting episode with difficulty: {difficulty}", file=sys.stderr)
        
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
        
        total_reward = 0.0
        step = 0
        api_call_count = 0
        
        # Episode loop - MAKE LLM CALLS FOR EVERY ACTION
        print(f"[INFO] Beginning episode loop (max {max_steps} steps)", file=sys.stderr)
        while step < max_steps:
            try:
                # GET ACTION FROM LLM - THIS IS THE CRITICAL PART
                action = self._call_llm_for_action(obs, difficulty, step, env_info)
                api_call_count += 1
                print(f"[SUCCESS] API call #{api_call_count}: Got action {action}", file=sys.stderr)
                
            except Exception as e:
                print(f"[ERROR] LLM call failed at step {step}: {e}", file=sys.stderr)
                raise  # Don't silently fail, re-raise so validator knows there's a problem
            
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
                print(f"[INFO] Episode terminated at step {step}", file=sys.stderr)
                break
        
        print(f"[INFO] Episode completed: {step} steps, {api_call_count} API calls made", file=sys.stderr)
        
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
            "api_calls": api_call_count,
            "env_state": env.state(),
        }
        
        return final_score, info
    
    def _call_llm_for_action(
        self,
        obs: np.ndarray,
        difficulty: str,
        step: int,
        env_info: Dict[str, Any],
    ) -> int:
        """
        MANDATORY: Call LLM API through validator's LiteLLM proxy.
        This is the ONLY place we get actions from.
        
        Args:
            obs: Current observation
            difficulty: Current task difficulty
            step: Current step number
            env_info: Environment info
        
        Returns:
            Action index (0-7)
        """
        # Prepare prompt
        prompt = self._prepare_llm_prompt(obs, difficulty, step, env_info)
        
        # PRINT BEFORE MAKING THE CALL
        print(f"[DEBUG] Making API call to: {self.api_base_url}", file=sys.stderr)
        print(f"[DEBUG] Using model: {self.model_name}", file=sys.stderr)
        print(f"[DEBUG] Step: {step}, Difficulty: {difficulty}", file=sys.stderr)
        
        # MAKE THE API CALL to validator's proxy
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are controlling an autonomous drone in a disaster rescue simulation. "
                               "Respond with ONLY a single number 0-7 representing the action.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            max_tokens=5,
            temperature=0.7,
        )
        
        # Parse response
        try:
            action_str = response.choices[0].message.content.strip()
            action = int(action_str)
            action = np.clip(action, 0, 7)
            print(f"[DEBUG] LLM response parsed: action={action}", file=sys.stderr)
            return action
        except Exception as e:
            print(f"[ERROR] Failed to parse LLM response: {response.choices[0].message.content}", file=sys.stderr)
            raise ValueError(f"Invalid LLM response format: {e}")
    
    def _prepare_llm_prompt(
        self,
        obs: np.ndarray,
        difficulty: str,
        step: int,
        env_info: Dict[str, Any],
    ) -> str:
        """
        Prepare prompt for LLM.
        """
        battery = env_info.get('battery', 0)
        victims_rescued = env_info.get('victims_rescued', 0)
        total_victims = env_info.get('total_victims', 5)
        hazards_nearby = np.sum(obs[:,:,2]) > 0
        victims_visible = np.sum(obs[:,:,1]) > 0
        
        prompt = f"""Disaster Rescue Simulation
Difficulty: {difficulty}
Step: {step}

Current Status:
- Battery: {battery}
- Victims Rescued: {victims_rescued}/{total_victims}

Available Actions:
0=North, 1=Northeast, 2=East, 3=Southeast
4=South, 5=Southwest, 6=West, 7=Northwest

Situation:
- Victims nearby: {'YES' if victims_visible else 'NO'}
- Hazards nearby: {'YES' if hazards_nearby else 'NO'}

Choose best action (respond with ONLY the number 0-7):"""
        
        return prompt
    
    def run_all_tasks(self, seed: Optional[int] = None) -> Dict[str, float]:
        """
        Run all three difficulty levels sequentially.
        EACH step makes an LLM API call.
        
        Args:
            seed: Random seed for reproducibility
        
        Returns:
            Dictionary mapping difficulty -> score
        """
        results = {}
        
        print("[INFO] ========================================", file=sys.stderr)
        print("[INFO] Starting inference with LLM API calls", file=sys.stderr)
        print(f"[INFO] Using API endpoint: {self.api_base_url}", file=sys.stderr)
        print(f"[INFO] Using model: {self.model_name}", file=sys.stderr)
        print("[INFO] ========================================", file=sys.stderr)
        
        for difficulty in get_all_difficulties():
            print(f"[INFO] ========================================", file=sys.stderr)
            print(f"[INFO] Running {difficulty.upper()} task", file=sys.stderr)
            print(f"[INFO] ========================================", file=sys.stderr)
            
            try:
                score, info = self.run_episode(
                    difficulty=difficulty,
                    seed=seed,
                )
                results[difficulty] = score
                api_calls = info.get('api_calls', 0)
                print(
                    f"[INFO] {difficulty.upper()} COMPLETED - Score: {score:.4f}, API Calls: {api_calls}",
                    file=sys.stderr,
                )
            except Exception as e:
                print(f"[CRITICAL] {difficulty.upper()} FAILED: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc(file=sys.stderr)
                results[difficulty] = 0.0
        
        self.results = results
        return results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all runs."""
        if not self.results:
            return {"error": "No results yet"}
        
        scores = list(self.results.values())
        return {
            "run_id": self.run_id,
            "model": self.model_name,
            "api_base_url": self.api_base_url,
            "results": self.results,
            "mean_score": float(np.mean(scores)) if scores else 0.0,
            "min_score": float(np.min(scores)) if scores else 0.0,
            "max_score": float(np.max(scores)) if scores else 0.0,
        }


def main():
    """
    Main entry point for inference.
    
    REQUIRED Environment Variables (provided by validator):
    - API_BASE_URL: The LiteLLM proxy endpoint (REQUIRED)
    - API_KEY: The LiteLLM proxy API key (REQUIRED)
    - MODEL_NAME: The model to use (default: gpt-3.5-turbo)
    """
    try:
        print("[INFO] Starting inference runner...", file=sys.stderr)
        
        # Create runner (will fail immediately if credentials not set)
        runner = InferenceRunner()
        
        # Run all tasks with LLM
        print("[INFO] Running all tasks...", file=sys.stderr)
        results = runner.run_all_tasks()
        
        # Print summary
        summary = runner.get_summary()
        print("[INFO] ========================================", file=sys.stderr)
        print("[INFO] INFERENCE COMPLETE!", file=sys.stderr)
        print("[INFO] ========================================", file=sys.stderr)
        print(json.dumps(summary, indent=2), file=sys.stderr)
        
        return 0
    
    except Exception as e:
        print(f"[FATAL ERROR] Inference failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)