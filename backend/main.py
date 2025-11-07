from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import uvicorn
import sys
import os
from pathlib import Path
import threading
import uuid
from typing import Optional

# Add parent directory to path to import agent2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI(title="Web Navigator Agent API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active sessions
sessions = {}

class TaskRequest(BaseModel):
    goal: str

class TaskResponse(BaseModel):
    session_id: str
    message: str
    snapshot_url: str

@app.get("/")
def read_root():
    return {"message": "Web Navigator Agent API", "status": "running"}

@app.post("/api/task", response_model=TaskResponse)
async def create_task(task: TaskRequest):
    """
    Receive a task from the frontend and execute the agent
    """
    try:
        # Generate unique session ID
        session_id = str(uuid.uuid4())
        
        # Store session info
        sessions[session_id] = {
            "goal": task.goal,
            "status": "running",
            "screenshots": []
        }
        
        # Run agent in background thread
        def run_agent():
            try:
                from agent2 import run_agent_task
                screenshots = run_agent_task(task.goal)
                sessions[session_id]["screenshots"] = screenshots
                sessions[session_id]["status"] = "completed"
            except Exception as e:
                sessions[session_id]["status"] = "failed"
                sessions[session_id]["error"] = str(e)
                print(f"Agent error: {e}")
        
        thread = threading.Thread(target=run_agent, daemon=True)
        thread.start()
        
        return TaskResponse(
            session_id=session_id,
            message="Task started successfully",
            snapshot_url=f"/snapshots/{session_id}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/session/{session_id}")
async def get_session_status(session_id: str):
    """
    Get the status of a session
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    return {
        "session_id": session_id,
        "status": session["status"],
        "goal": session["goal"],
        "screenshots": session.get("screenshots", []),
        "error": session.get("error")
    }

@app.get("/api/snapshots/{session_id}")
async def get_snapshots(session_id: str):
    """
    Get list of screenshots for a session
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Get all screenshots from the screenshots folder
    screenshots_dir = Path(__file__).parent.parent / "screenshots"
    
    if not screenshots_dir.exists():
        return {"screenshots": [], "message": "No screenshots directory found"}
    
    screenshot_files = sorted(
        [f.name for f in screenshots_dir.glob("step_*.png")],
        key=lambda x: int(x.split("_")[1].split(".")[0]) if "_" in x else 0
    )
    
    return {
        "session_id": session_id,
        "status": sessions[session_id]["status"],
        "screenshots": screenshot_files
    }

@app.get("/api/screenshot/{filename}")
async def get_screenshot(filename: str):
    """
    Serve a specific screenshot file
    """
    screenshots_dir = Path(__file__).parent.parent / "screenshots"
    file_path = screenshots_dir / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Screenshot not found")
    
    return FileResponse(file_path, media_type="image/png")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

