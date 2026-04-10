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
    """
    
    def __init__(self):
        """
        Initialize inference runner.
        MUST use ONLY validator-provided API_BASE_URL and API_KEY.
        """
        # Read DIRECTLY from os.environ - NO fallbacks, NO defaults
        api_base_url = os.environ["API_BASE_URL"]
        api_key = os.environ["API_KEY"]
        model_name = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")
        
        self.api_base_url = api_base_url
        self.api_key = api_key
        self.model_name = model_name
        
        # Initialize logger
        self.logger = StructuredLogger(use_stdout=True)
        
        # Verify we have the required credentials
        print(f"[INFO] API_BASE_URL: {self.api_base_url}", file=sys.stderr)
        print(f"[INFO] API_KEY length: {len(self.api_key)}", file=sys.stderr)
        print(f"[INFO] MODEL_NAME: {self.model_name}", file=sys.stderr)
        
        # Initialize OpenAI client with VALIDATOR'S credentials ONLY
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=api_key,
                base_url=api_base_url,
            )
            print(f"[SUCCESS] OpenAI client initialized with validator credentials", file=sys.stderr)
        except Exception as e:
            print(f"[FATAL] Failed to initialize OpenAI client: {e}", file=sys.stderr)
            raise
        
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
        print(f"[EPISODE] Starting {difficulty} episode", file=sys.stderr)
        
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
        
        # Episode loop - MAKE LLM API CALL FOR EVERY ACTION
        print(f"[EPISODE] Beginning step loop...", file=sys.stderr)
        while step < max_steps:
            # CALL LLM FOR ACTION - THIS IS REQUIRED
            try:
                action = self._get_llm_action(obs, difficulty, step, env_info)
                api_call_count += 1
                print(f"[API_CALL] #{api_call_count}: Action={action}", file=sys.stderr)
            except Exception as e:
                print(f"[FATAL] LLM API call failed at step {step}: {e}", file=sys.stderr)
                raise
            
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
                print(f"[EPISODE] Episode terminated at step {step}", file=sys.stderr)
                break
        
        print(f"[EPISODE] Completed with {api_call_count} API calls", file=sys.stderr)
        
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
    
    def _get_llm_action(
        self,
        obs: np.ndarray,
        difficulty: str,
        step: int,
        env_info: Dict[str, Any],
    ) -> int:
        """
        Get action from LLM using validator's LiteLLM proxy.
        
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
        
        # DEBUG: Show we're using the correct endpoint
        print(f"[DEBUG] API endpoint: {self.api_base_url}", file=sys.stderr)
        print(f"[DEBUG] Model: {self.model_name}", file=sys.stderr)
        
        # MAKE API CALL to validator's LiteLLM proxy
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are an autonomous drone controller in a disaster rescue simulation. "
                               "Respond with ONLY a single number 0-7.",
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
        action_str = response.choices[0].message.content.strip()
        action = int(action_str)
        return np.clip(action, 0, 7)
    
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
        hazards_nearby = np.sum(obs[:,:,2]) > 0 if len(obs.shape) > 2 and obs.shape[2] > 2 else False
        victims_visible = np.sum(obs[:,:,1]) > 0 if len(obs.shape) > 2 and obs.shape[2] > 1 else False
        
        prompt = f"""Disaster Rescue Simulation
Difficulty: {difficulty}
Step: {step}

Current Status:
Battery: {battery}
Victims Rescued: {victims_rescued}/{total_victims}

Actions:
0=North, 1=Northeast, 2=East, 3=Southeast
4=South, 5=Southwest, 6=West, 7=Northwest

Situation:
Victims Visible: {victims_visible}
Hazards Nearby: {hazards_nearby}

Choose action (0-7):"""
        
        return prompt
    
    def run_all_tasks(self, seed: Optional[int] = None) -> Dict[str, float]:
        """
        Run all three difficulty levels sequentially.
        Each action makes an API call to validator's LiteLLM proxy.
        
        Args:
            seed: Random seed for reproducibility
        
        Returns:
            Dictionary mapping difficulty -> score
        """
        results = {}
        
        print("[INFO] ========================================", file=sys.stderr)
        print("[INFO] STARTING INFERENCE", file=sys.stderr)
        print(f"[INFO] Using LiteLLM proxy: {self.api_base_url}", file=sys.stderr)
        print(f"[INFO] Model: {self.model_name}", file=sys.stderr)
        print("[INFO] ========================================", file=sys.stderr)
        
        for difficulty in get_all_difficulties():
            print(f"[INFO] ========================================", file=sys.stderr)
            print(f"[INFO] DIFFICULTY: {difficulty.upper()}", file=sys.stderr)
            print(f"[INFO] ========================================", file=sys.stderr)
            
            try:
                score, info = self.run_episode(
                    difficulty=difficulty,
                    seed=seed,
                )
                results[difficulty] = score
                api_calls = info.get('api_calls', 0)
                print(
                    f"[RESULT] {difficulty.upper()}: Score={score:.4f}, API_Calls={api_calls}",
                    file=sys.stderr,
                )
            except Exception as e:
                print(f"[FATAL] {difficulty.upper()} FAILED: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc(file=sys.stderr)
                raise
        
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
    - API_BASE_URL: LiteLLM proxy endpoint
    - API_KEY: LiteLLM proxy API key
    - MODEL_NAME: Model to use (optional, default: gpt-3.5-turbo)
    
    This script will ONLY use validator-provided credentials.
    No fallbacks, no defaults for API_BASE_URL and API_KEY.
    """
    try:
        print("[INFO] Starting inference...", file=sys.stderr)
        
        # Create runner - will fail if validator hasn't provided credentials
        runner = InferenceRunner()
        
        # Run all tasks
        results = runner.run_all_tasks()
        
        # Print summary
        summary = runner.get_summary()
        print("[INFO] ========================================", file=sys.stderr)
        print("[INFO] INFERENCE COMPLETE!", file=sys.stderr)
        print("[INFO] ========================================", file=sys.stderr)
        print(json.dumps(summary, indent=2), file=sys.stderr)
        
        return 0
    
    except KeyError as e:
        print(f"[FATAL] Missing required environment variable: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"[FATAL] Inference failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
