from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
import os
import json
from datetime import datetime
import uuid

# Import your environment and agents
from environment.disaster_env import DisasterRescueEnv
from agents.baseline_agent import get_agent
from utils.graders import get_grader
from utils.logger import StructuredLogger

app = FastAPI(
    title="Disaster Rescue RL Environment",
    description="Autonomous Drone Search & Rescue Simulation",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active sessions
sessions = {}
logger = StructuredLogger()

# =====================
# Root Redirect
# =====================
@app.get("/")
async def root():
    """Redirect to API docs"""
    return RedirectResponse(url="/docs")

# =====================
# Health & Info Endpoints
# =====================
@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "message": "Disaster Rescue RL Environment API is running"
    }

@app.get("/info")
async def info():
    """Get API information and capabilities"""
    return {
        "api_name": "Disaster Rescue RL Environment",
        "version": "1.0.0",
        "description": "Autonomous Drone Search & Rescue Simulation",
        "supported_difficulties": ["easy", "medium", "hard"],
        "supported_agents": ["random", "exploration", "grid_search", "greedy", "hybrid"],
        "action_space": 8,
        "observation_shape": [64, 64, 5],
        "endpoints": [
            {"method": "GET", "path": "/health", "description": "Health check"},
            {"method": "GET", "path": "/info", "description": "API info"},
            {"method": "POST", "path": "/reset", "description": "Create session"},
            {"method": "POST", "path": "/step", "description": "Take action"},
            {"method": "GET", "path": "/state/{session_id}", "description": "Get state"},
            {"method": "POST", "path": "/evaluate", "description": "Evaluate agent"},
            {"method": "GET", "path": "/sessions", "description": "List sessions"},
        ]
    }

# =====================
# Environment Endpoints
# =====================
@app.post("/reset")
async def reset(difficulty: str = "easy", seed: int = None):
    """Reset environment and create new session"""
    try:
        session_id = str(uuid.uuid4())
        env = DisasterRescueEnv(difficulty=difficulty, seed=seed)
        obs, info = env.reset(seed=seed)
        
        sessions[session_id] = {
            "env": env,
            "difficulty": difficulty,
            "step": 0,
            "observation": obs,
            "info": info,
            "history": []
        }
        
        return {
            "session_id": session_id,
            "difficulty": difficulty,
            "observation_shape": list(obs.shape),
            "action_space": 8,
            "initial_info": {
                "battery": info.get("battery", 1000),
                "total_victims": info.get("total_victims", 5),
                "hazards": info.get("hazards", 8)
            }
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/step")
async def step(session_id: str, action: int):
    """Take one step in the environment"""
    try:
        if session_id not in sessions:
            return {"error": "Session not found"}
        
        session = sessions[session_id]
        env = session["env"]
        obs, reward, terminated, truncated, info = env.step(action)
        
        session["observation"] = obs
        session["step"] += 1
        session["info"] = info
        
        return {
            "session_id": session_id,
            "step": session["step"],
            "action": action,
            "reward": float(reward),
            "terminated": terminated,
            "truncated": truncated,
            "info": {
                "battery": info.get("battery", 0),
                "victims_rescued": info.get("victims_rescued", 0),
                "total_victims": info.get("total_victims", 0),
            }
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/state/{session_id}")
async def get_state(session_id: str):
    """Get current environment state"""
    try:
        if session_id not in sessions:
            return {"error": "Session not found"}
        
        session = sessions[session_id]
        return {
            "session_id": session_id,
            "step": session["step"],
            "difficulty": session["difficulty"],
            "state": {
                "battery": session["info"].get("battery", 0),
                "victims_rescued": session["info"].get("victims_rescued", 0),
                "total_victims": session["info"].get("total_victims", 0),
            }
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/evaluate")
async def evaluate(agent_type: str = "greedy", difficulty: str = "easy", num_episodes: int = 3, seed: int = None):
    """Evaluate an agent"""
    try:
        env = DisasterRescueEnv(difficulty=difficulty, seed=seed)
        agent = get_agent(agent_type, env=env, seed=seed)
        grader = get_grader(difficulty)
        
        scores = []
        rewards = []
        
        for episode in range(num_episodes):
            obs, _ = env.reset(seed=seed)
            total_reward = 0
            done = False
            steps = 0
            
            while not done and steps < 50:
                action, _ = agent.predict(obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                steps += 1
                done = terminated or truncated
            
            score = grader.grade(env.state())
            scores.append(score)
            rewards.append(total_reward)
        
        return {
            "agent_type": agent_type,
            "difficulty": difficulty,
            "num_episodes": num_episodes,
            "scores": {
                "mean": float(sum(scores) / len(scores)),
                "std": float((sum((x - sum(scores)/len(scores))**2 for x in scores) / len(scores))**0.5),
                "min": float(min(scores)),
                "max": float(max(scores))
            },
            "rewards": {
                "mean": float(sum(rewards) / len(rewards)),
                "std": float((sum((x - sum(rewards)/len(rewards))**2 for x in rewards) / len(rewards))**0.5),
                "min": float(min(rewards)),
                "max": float(max(rewards))
            }
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/sessions")
async def list_sessions():
    """List all active sessions"""
    return {
        "total_sessions": len(sessions),
        "sessions": list(sessions.keys())
    }

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session"""
    if session_id in sessions:
        del sessions[session_id]
        return {"message": "Session deleted"}
    return {"error": "Session not found"}

# =====================
# Run Application
# =====================
if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host=host, port=port)