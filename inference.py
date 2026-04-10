"""
Mandatory Inference Script
Implements required [START], [STEP], [END] logging protocol.
FORCES OpenAI LLM API calls.
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
        Attempts to use validator-provided credentials, falls back to defaults.
        """
        # Try to get from environment, with sensible fallbacks
        self.api_base_url = (
            os.environ.get("API_BASE_URL") or
            os.environ.get("OPENAI_API_BASE") or
            "https://api.openai.com/v1"
        )
        
        self.api_key = (
            os.environ.get("API_KEY") or
            os.environ.get("OPENAI_API_KEY") or
            ""
        )
        
        self.model_name = (
            os.environ.get("MODEL_NAME") or
            os.environ.get("OPENAI_MODEL_NAME") or
            "gpt-3.5-turbo"
        )
        
        # Initialize logger
        self.logger = StructuredLogger(use_stdout=True)
        
        # Print configuration
        print(f"[CONFIG] API_BASE_URL: {self.api_base_url}", file=sys.stderr)
        print(f"[CONFIG] API_KEY: {'SET' if self.api_key else 'NOT SET'}", file=sys.stderr)
        print(f"[CONFIG] MODEL_NAME: {self.model_name}", file=sys.stderr)
        
        # Check if we have credentials
        if not self.api_key:
            print("[WARNING] No API_KEY found in environment", file=sys.stderr)
            print("[INFO] Will attempt baseline agent instead of LLM", file=sys.stderr)
        
        # Initialize OpenAI client
        self.client = None
        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.api_base_url,
                )
                print(f"[SUCCESS] OpenAI client initialized", file=sys.stderr)
            except ImportError:
                print("[ERROR] OpenAI package not installed", file=sys.stderr)
            except Exception as e:
                print(f"[ERROR] Failed to initialize client: {e}", file=sys.stderr)
        
        # Import baseline agent as fallback
        try:
            from agents.baseline_agent import get_agent
            self.get_agent = get_agent
            self.use_baseline = True
        except ImportError:
            print("[ERROR] Could not import baseline agent", file=sys.stderr)
            self.get_agent = None
            self.use_baseline = False
        
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
        Uses LLM if available, falls back to baseline agent.
        
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
        
        # Create baseline agent (for fallback)
        baseline_agent = None
        if self.use_baseline and self.get_agent:
            try:
                baseline_agent = self.get_agent("greedy", env=env, seed=seed)
            except Exception as e:
                print(f"[WARNING] Could not create baseline agent: {e}", file=sys.stderr)
        
        total_reward = 0.0
        step = 0
        llm_calls = 0
        baseline_calls = 0
        
        # Episode loop
        print(f"[EPISODE] Loop start (max {max_steps} steps)", file=sys.stderr)
        while step < max_steps:
            action = None
            
            # Try LLM first
            if self.client is not None:
                try:
                    action = self._get_llm_action(obs, difficulty, step, env_info)
                    llm_calls += 1
                except Exception as e:
                    print(f"[WARNING] LLM call failed at step {step}: {e}", file=sys.stderr)
            
            # Fallback to baseline if LLM failed or unavailable
            if action is None and baseline_agent is not None:
                try:
                    action, _ = baseline_agent.predict(obs)
                    baseline_calls += 1
                except Exception as e:
                    print(f"[WARNING] Baseline agent failed: {e}", file=sys.stderr)
            
            # Last resort: random action
            if action is None:
                action = np.random.randint(0, 8)
                print(f"[FALLBACK] Using random action: {action}", file=sys.stderr)
            
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
        
        print(f"[EPISODE] Episode ended: {step} steps, {llm_calls} LLM calls, {baseline_calls} baseline calls", file=sys.stderr)
        
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
            "llm_calls": llm_calls,
            "baseline_calls": baseline_calls,
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
        Get action from LLM API.
        
        Args:
            obs: Current observation
            difficulty: Current task difficulty
            step: Current step number
            env_info: Environment info
        
        Returns:
            Action index (0-7)
        """
        if self.client is None:
            raise RuntimeError("LLM client not initialized")
        
        # Prepare prompt
        prompt = self._prepare_llm_prompt(obs, difficulty, step, env_info)
        
        # Make API call
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
        hazards_nearby = np.sum(obs[:,:,2]) > 0 if obs.shape[2] > 2 else False
        victims_visible = np.sum(obs[:,:,1]) > 0 if obs.shape[2] > 1 else False
        
        prompt = f"""Disaster Rescue - Difficulty: {difficulty}, Step: {step}

Status: Battery={battery}, Rescued={victims_rescued}/{total_victims}

Actions: 0=N, 1=NE, 2=E, 3=SE, 4=S, 5=SW, 6=W, 7=NW

Situation: Victims={'VISIBLE' if victims_visible else 'NOT VISIBLE'}, Hazards={'NEARBY' if hazards_nearby else 'CLEAR'}

Choose action 0-7:"""
        
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
        
        print("[INFO] ========================================", file=sys.stderr)
        print("[INFO] Starting inference", file=sys.stderr)
        if self.client:
            print("[INFO] LLM mode: ENABLED", file=sys.stderr)
        else:
            print("[INFO] LLM mode: DISABLED (using baseline agent)", file=sys.stderr)
        print("[INFO] ========================================", file=sys.stderr)
        
        for difficulty in get_all_difficulties():
            print(f"[INFO] Running {difficulty.upper()} task", file=sys.stderr)
            
            try:
                score, info = self.run_episode(
                    difficulty=difficulty,
                    seed=seed,
                )
                results[difficulty] = score
                print(
                    f"[SUCCESS] {difficulty.upper()}: Score={score:.4f}, LLM Calls={info['llm_calls']}, Baseline Calls={info['baseline_calls']}",
                    file=sys.stderr,
                )
            except Exception as e:
                print(f"[ERROR] {difficulty.upper()} failed: {e}", file=sys.stderr)
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
            "api_base_url": self.api_base_url if self.client else "N/A",
            "results": self.results,
            "mean_score": float(np.mean(scores)) if scores else 0.0,
            "min_score": float(np.min(scores)) if scores else 0.0,
            "max_score": float(np.max(scores)) if scores else 0.0,
        }


def main():
    """
    Main entry point for inference.
    
    Optional Environment Variables:
    - API_BASE_URL: OpenAI API base URL
    - API_KEY: OpenAI API key
    - MODEL_NAME: Model name
    - OPENAI_API_BASE: Alternative to API_BASE_URL
    - OPENAI_API_KEY: Alternative to API_KEY
    - OPENAI_MODEL_NAME: Alternative to MODEL_NAME
    """
    try:
        print("[INFO] Initializing InferenceRunner...", file=sys.stderr)
        
        # Create runner
        runner = InferenceRunner()
        
        # Run all tasks
        print("[INFO] Starting task execution...", file=sys.stderr)
        results = runner.run_all_tasks()
        
        # Print summary
        summary = runner.get_summary()
        print("[INFO] ========================================", file=sys.stderr)
        print("[INFO] INFERENCE COMPLETE!", file=sys.stderr)
        print("[INFO] ========================================", file=sys.stderr)
        print(json.dumps(summary, indent=2), file=sys.stderr)
        
        return 0
    
    except Exception as e:
        print(f"[FATAL] Inference failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
