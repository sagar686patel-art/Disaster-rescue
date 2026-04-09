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
        Initialize inference runner with environment variables.
        REQUIRES: API_BASE_URL and API_KEY environment variables.
        """
        # CRITICAL: Get from environment variables - validator will inject these
        self.api_base_url = os.environ.get("API_BASE_URL")
        self.api_key = os.environ.get("API_KEY")
        self.model_name = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")
        
        # Validate that required environment variables are set
        if not self.api_base_url:
            raise ValueError("API_BASE_URL environment variable is required")
        if not self.api_key:
            raise ValueError("API_KEY environment variable is required")
        
        print(f"[INFO] Initialized with API_BASE_URL: {self.api_base_url}", file=sys.stderr)
        print(f"[INFO] Using model: {self.model_name}", file=sys.stderr)
        
        # Initialize logger
        self.logger = StructuredLogger(use_stdout=True)
        
        # Initialize OpenAI client with REQUIRED environment variables
        # NOTE: Do NOT pass proxies parameter - use base_url instead
        try:
            from openai import OpenAI
            
            # Simple initialization - only pass api_key and base_url
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base_url
            )
            print(f"[INFO] OpenAI client initialized successfully", file=sys.stderr)
            print(f"[INFO] Base URL: {self.api_base_url}", file=sys.stderr)
        except TypeError as e:
            # Handle OpenAI version incompatibility
            print(f"[ERROR] OpenAI client initialization failed: {e}", file=sys.stderr)
            print(f"[ERROR] This may be due to OpenAI library version mismatch", file=sys.stderr)
            raise
        except Exception as e:
            print(f"[CRITICAL ERROR] Failed to initialize OpenAI client: {e}", file=sys.stderr)
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
        Run a single episode using LLM for action selection.
        
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
        
        total_reward = 0.0
        step = 0
        llm_call_count = 0
        
        # Episode loop - ALWAYS use LLM
        while step < max_steps:
            try:
                # Get action from LLM (MANDATORY)
                action = self._get_llm_action(obs, difficulty, step, env_info)
                llm_call_count += 1
            except Exception as e:
                print(f"[ERROR] LLM action selection failed at step {step}: {e}", file=sys.stderr)
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
                break
        
        # Grade the episode
        grader = get_grader(difficulty)
        final_score = grader.grade(env.state())
        
        # Log episode end (mandatory format)
        episode_logger.end(final_score=final_score)
        
        print(f"[INFO] Episode completed: difficulty={difficulty}, steps={step}, llm_calls={llm_call_count}, score={final_score:.4f}", file=sys.stderr)
        
        info = {
            "difficulty": difficulty,
            "steps": step,
            "llm_calls": llm_call_count,
            "total_reward": total_reward,
            "score": final_score,
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
        THIS FUNCTION MUST MAKE API CALLS through the provided API_BASE_URL.
        
        Args:
            obs: Current observation
            difficulty: Current task difficulty
            step: Current step number
            env_info: Environment info
        
        Returns:
            Action index (0-7)
        """
        if self.client is None:
            raise RuntimeError("OpenAI client not initialized")
        
        try:
            # Prepare prompt
            prompt = self._prepare_llm_prompt(obs, difficulty, step, env_info)
            
            # CRITICAL: Call LLM through provided API credentials
            print(f"[DEBUG] Making LLM call at step {step}...", file=sys.stderr)
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an AI controlling an autonomous drone in a disaster rescue simulation. "
                                   "Choose the best action given the current state. "
                                   "Respond with ONLY a single number 0-7, nothing else.",
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
            
            # Extract just the number
            try:
                action = int(action_str)
            except ValueError:
                # If response contains extra text, try to extract the number
                import re
                numbers = re.findall(r'\d', action_str)
                if numbers:
                    action = int(numbers[0])
                else:
                    print(f"[WARNING] Could not parse LLM response: {action_str}", file=sys.stderr)
                    action = np.random.randint(0, 8)
            
            action = int(np.clip(action, 0, 7))
            print(f"[DEBUG] LLM returned action: {action}", file=sys.stderr)
            return action
        
        except Exception as e:
            print(f"[ERROR] LLM call failed at step {step}: {e}", file=sys.stderr)
            print(f"[ERROR] Exception type: {type(e).__name__}", file=sys.stderr)
            raise
    
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
        # Analyze observation channels
        try:
            agent_pos = np.sum(obs[:,:,0]) > 0
            victims_visible = np.sum(obs[:,:,1]) > 0
            hazards_nearby = np.sum(obs[:,:,2]) > 0
            resources_available = np.sum(obs[:,:,3]) > 0
        except Exception as e:
            print(f"[WARNING] Error analyzing observation: {e}", file=sys.stderr)
            agent_pos = victims_visible = hazards_nearby = resources_available = False
        
        prompt = f"""Current State - Step {step} ({difficulty.upper()}):
Battery: {env_info.get('battery', 'Unknown')}
Victims: {env_info.get('victims_rescued', 0)}/{env_info.get('total_victims', 1)} rescued
Observations: Victims visible={victims_visible}, Hazards={hazards_nearby}, Resources={resources_available}

Actions: 0=N, 1=NE, 2=E, 3=SE, 4=S, 5=SW, 6=W, 7=NW

Choose best action (respond with single digit 0-7 only):"""
        
        return prompt
    
    def run_all_tasks(self, seed: Optional[int] = None) -> Dict[str, float]:
        """
        Run all three difficulty levels sequentially with LLM.
        
        Args:
            seed: Random seed for reproducibility
        
        Returns:
            Dictionary mapping difficulty -> score
        """
        results = {}
        
        print("[INFO] ========== STARTING INFERENCE ==========", file=sys.stderr)
        print(f"[INFO] API_BASE_URL: {self.api_base_url}", file=sys.stderr)
        print(f"[INFO] Model: {self.model_name}", file=sys.stderr)
        print(f"[INFO] Run ID: {self.run_id}", file=sys.stderr)
        print("[INFO] ========================================", file=sys.stderr)
        
        for difficulty in get_all_difficulties():
            print(f"\n[INFO] Running {difficulty.upper()} task...", file=sys.stderr)
            
            try:
                score, info = self.run_episode(
                    difficulty=difficulty,
                    seed=seed,
                )
                results[difficulty] = score
                print(f"[SUCCESS] {difficulty.upper()}: Score={score:.4f}, LLM calls={info['llm_calls']}", file=sys.stderr)
            except Exception as e:
                print(f"[ERROR] {difficulty} task failed: {e}", file=sys.stderr)
                import traceback
                print(traceback.format_exc(), file=sys.stderr)
                raise
        
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
            "api_base_url": self.api_base_url,
            "results": self.results,
            "mean_score": float(np.mean(scores)) if scores else 0.0,
            "min_score": float(np.min(scores)) if scores else 0.0,
            "max_score": float(np.max(scores)) if scores else 0.0,
        }


def main():
    """
    Main entry point for inference.
    
    REQUIRED Environment Variables (injected by validator):
    - API_BASE_URL: LiteLLM proxy URL (e.g., https://litellm.sclr.ac)
    - API_KEY: Authentication key for proxy
    - MODEL_NAME: (optional) Model name, default: gpt-3.5-turbo
    - SEED: (optional) Random seed
    """
    try:
        # Validate environment
        if not os.environ.get("API_BASE_URL"):
            raise EnvironmentError("API_BASE_URL not found in environment")
        if not os.environ.get("API_KEY"):
            raise EnvironmentError("API_KEY not found in environment")
        
        # Get optional parameters
        seed = int(os.environ.get("SEED", "0")) if os.environ.get("SEED") else None
        
        print("[INFO] Environment variables validation passed", file=sys.stderr)
        
        # Create runner (initializes OpenAI client)
        runner = InferenceRunner()
        
        # Run all tasks with LLM
        results = runner.run_all_tasks(seed=seed)
        
        # Print summary
        summary = runner.get_summary()
        print("\n[INFO] ========== FINAL SUMMARY ==========", file=sys.stderr)
        print(json.dumps(summary, indent=2), file=sys.stderr)
        print("[INFO] ===================================\n", file=sys.stderr)
        
        return 0
    
    except EnvironmentError as e:
        print(f"[CRITICAL ERROR] Environment error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"[CRITICAL ERROR] {e}", file=sys.stderr)
        import traceback
        print(traceback.format_exc(), file=sys.stderr)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
