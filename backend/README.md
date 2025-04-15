# YouTube Comment Sentiment Analysis API

This is the backend API for a YouTube comment sentiment analysis tool.

## Features

- Extract comments from YouTube videos
- Analyze comment sentiment using VADER and TextBlob
- Get detailed sentiment analysis reports
- Provide video metadata

## Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file based on `.env.example` and add your YouTube API key
4. Run the server:
   ```
   python main.py
   ```
   Or with uvicorn directly:
   ```
   uvicorn main:app --reload
   ```

## API Endpoints

- `GET /`: Root endpoint to verify the API is running
- `POST /analyze`: Analyze YouTube comments
  - Request body: `{"url": "youtube_video_url", "max_comments": 100}`
  - Returns sentiment analysis results and video information

## Getting a YouTube API Key

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project
3. Enable the YouTube Data API v3
4. Create API credentials
5. Add the API key to your `.env` file

## Requirements

Python 3.8+ and packages listed in `requirements.txt`
