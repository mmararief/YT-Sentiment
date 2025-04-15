@echo off
echo Starting YouTube Comment Sentiment Analysis application...

start cmd /k "cd backend && echo Starting Backend... && python -m venv env && env\Scripts\activate && pip install -r requirements.txt && python main.py"
start cmd /k "cd frontend && echo Starting Frontend... && npm install && npm run dev"

echo Services starting in new windows...
echo Backend will be available at http://localhost:8000
echo Frontend will be available at http://localhost:3000
echo.
echo IMPORTANT: Make sure to set up your YouTube API key in backend/.env before using the application! 