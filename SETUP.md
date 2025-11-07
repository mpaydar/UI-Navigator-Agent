# Web Navigator Agent - Setup Guide

A full-stack web automation agent with FastAPI backend and Next.js frontend.

## Architecture

- **Backend**: FastAPI (Python) - Executes the automation agent
- **Frontend**: Next.js (React/TypeScript) - User interface
- **Agent**: Playwright + LangGraph + GPT-4 Vision

## Prerequisites

- Python 3.12+
- Node.js 18+
- OpenAI API Key

## Setup Instructions

### 1. Environment Variables

Create a `.env` file in the root directory:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### 2. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Install Python dependencies (make sure .venv is activated)
pip install -r requirements.txt

# Also install the main project dependencies (from root)
cd ..
pip install langgraph langchain-core python-dotenv playwright openai

# Install Playwright browsers
playwright install

# Run the backend
cd backend
python main.py
```

Backend will start at: `http://localhost:8000`

### 3. Frontend Setup

```bash
# Navigate to frontend directory
cd front-end

# Install dependencies
npm install

# Run the development server
npm run dev
```

Frontend will start at: `http://localhost:3000`

## Usage

1. **Start the Backend**:
   ```bash
   cd backend
   python main.py
   ```

2. **Start the Frontend** (in a new terminal):
   ```bash
   cd front-end
   npm run dev
   ```

3. **Use the Application**:
   - Open `http://localhost:3000` in your browser
   - Enter your goal (e.g., "Create a new project in Linear")
   - Click "Start Agent"
   - You'll be redirected to the snapshots page where you can see:
     - Real-time execution status
     - Step-by-step screenshots
     - Click on any screenshot to view it full-size

## API Endpoints

### Backend (http://localhost:8000)

- `POST /api/task` - Submit a new task
  ```json
  {
    "goal": "Your task description"
  }
  ```

- `GET /api/session/{session_id}` - Get session status

- `GET /api/snapshots/{session_id}` - Get list of screenshots for a session

- `GET /api/screenshot/{filename}` - Get a specific screenshot image

### Frontend Routes

- `/` - Home page with task input
- `/snapshots/[sessionId]` - View execution results and screenshots

## Features

✅ **Full-stack integration** - FastAPI backend + Next.js frontend  
✅ **Real-time updates** - Auto-refresh while agent is running  
✅ **Screenshot gallery** - View all execution steps  
✅ **Full-size preview** - Click to enlarge any screenshot  
✅ **Session management** - Track multiple execution sessions  
✅ **Loading states** - Visual feedback during processing  
✅ **Error handling** - Graceful error messages  

## Project Structure

```
UI-Navigator-Agent/
├── backend/
│   ├── main.py              # FastAPI application
│   └── requirements.txt     # Backend dependencies
├── front-end/
│   ├── app/
│   │   ├── page.tsx        # Home page
│   │   └── snapshots/
│   │       └── [sessionId]/
│   │           └── page.tsx # Snapshots viewer
│   └── package.json
├── agent2.py                # Main agent logic
├── screenshots/             # Generated screenshots
├── .env                     # Environment variables
└── SETUP.md                # This file
```

## Troubleshooting

### Backend not starting
- Make sure Python 3.12+ is installed
- Check that all dependencies are installed
- Verify `.env` file has OPENAI_API_KEY

### Frontend can't connect to backend
- Ensure backend is running on port 8000
- Check CORS settings in `backend/main.py`
- Verify fetch URLs in frontend code

### No screenshots appearing
- Check that `screenshots/` directory exists
- Verify Playwright is properly installed
- Check backend console for agent execution logs

### Agent execution fails
- Ensure OpenAI API key is valid
- Check Playwright browsers are installed: `playwright install`
- Review backend console for error messages

## Development

### Backend Hot Reload
FastAPI will auto-reload when you make changes to `backend/main.py`

### Frontend Hot Reload
Next.js will auto-reload when you make changes to any files in `front-end/app/`

## Next Steps

- [ ] Add authentication
- [ ] Implement session history
- [ ] Add download all screenshots feature
- [ ] Add real-time streaming updates
- [ ] Deploy to production

## License

MIT

