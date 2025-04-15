from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any, Union
import googleapiclient.discovery
import googleapiclient.errors
import os
from dotenv import load_dotenv
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import json
from fastapi.responses import JSONResponse
from pytube import YouTube
import urllib.parse
import pickle
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import string
from sklearn.pipeline import Pipeline

# Download NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

# Load environment variables
load_dotenv()

app = FastAPI(title="YouTube Comment Sentiment Analysis API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model paths
MODEL_DIR = "models"
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, "best_model_svm_tfidf.pkl")

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Global variables for models
svm_model = None

# Text preprocessing functions
def preprocess_text(text):
    """Preprocess text for TF-IDF vectorization"""
    if text is None or not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    
    return ' '.join(tokens)

# Input models
class YouTubeURL(BaseModel):
    url: str
    max_comments: Optional[int] = 100
    use_svm_model: Optional[bool] = True

class ModelUpload(BaseModel):
    model_name: str
    description: Optional[str] = None

# Response models
class Comment(BaseModel):
    text: str
    author: str
    published_at: str
    likes: int
    sentiment_score: float
    sentiment_label: str

class SentimentSummary(BaseModel):
    positive: float
    negative: float
    neutral: float

class SentimentResponse(BaseModel):
    comments: List[Comment]
    summary: SentimentSummary
    video_info: Dict[str, Any]
    model_used: str

@app.on_event("startup")
async def startup_event():
    """Load ML models on startup"""
    global svm_model
    
    # Check if default model exists, if not create a placeholder
    if not os.path.exists(DEFAULT_MODEL_PATH):
        print(f"Warning: Default model not found at {DEFAULT_MODEL_PATH}")
        print(f"Looking for model in: {os.path.abspath(DEFAULT_MODEL_PATH)}")
    else:
        try:
            print(f"Loading SVM model from: {os.path.abspath(DEFAULT_MODEL_PATH)}")
            # Load the pipeline model
            svm_model = joblib.load(DEFAULT_MODEL_PATH)
            print(f"Successfully loaded SVM model: {type(svm_model)}")
            
            # Verify it's a pipeline
            if hasattr(svm_model, 'named_steps'):
                print("Model is a pipeline with vectorizer included")
            else:
                print("Warning: Model is not a pipeline. Please ensure you're using a pipeline model.")
                svm_model = None
        except Exception as e:
            print(f"Error loading SVM model: {str(e)}")
            print(f"Stack trace: ", e)
            svm_model = None

# Helper functions
def extract_video_id(url: str) -> str:
    """Extract video ID from YouTube URL"""
    # For URLs like: https://www.youtube.com/watch?v=VIDEO_ID
    query_pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(query_pattern, url)
    if match:
        return match.group(1)
    
    # For shortened URLs like: https://youtu.be/VIDEO_ID
    short_pattern = r"youtu\.be\/([0-9A-Za-z_-]{11})"
    match = re.search(short_pattern, url)
    if match:
        return match.group(1)
    
    raise ValueError("Could not extract video ID from URL")

def get_video_info(video_id: str) -> Dict[str, Any]:
    """Get basic information about a YouTube video"""
    try:
        # Initialize YouTube API
        api_key = os.getenv("YOUTUBE_API_KEY")
        if not api_key:
            # Fallback to pytube if API key is not available
            yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")
            return {
                "title": yt.title,
                "channel": yt.author,
                "thumbnail": yt.thumbnail_url,
                "views": yt.views,
                "video_id": video_id
            }
        
        youtube = googleapiclient.discovery.build(
            "youtube", "v3", developerKey=api_key
        )
        
        # Get video details
        request = youtube.videos().list(
            part="snippet,statistics",
            id=video_id
        )
        response = request.execute()
        
        if not response["items"]:
            raise HTTPException(status_code=404, detail="Video not found")
        
        video = response["items"][0]
        snippet = video["snippet"]
        statistics = video["statistics"]
        
        return {
            "title": snippet["title"],
            "channel": snippet["channelTitle"],
            "thumbnail": snippet["thumbnails"]["high"]["url"],
            "views": statistics.get("viewCount", 0),
            "video_id": video_id
        }
    except Exception as e:
        # Fallback to pytube if API fails
        try:
            yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")
            return {
                "title": yt.title,
                "channel": yt.author,
                "thumbnail": yt.thumbnail_url,
                "views": yt.views,
                "video_id": video_id
            }
        except:
            raise HTTPException(status_code=500, detail=f"Error fetching video info: {str(e)}")

def get_comments(video_id: str, max_results: int = 100) -> List[Dict[str, Any]]:
    """Get comments from a YouTube video"""
    try:
        # Initialize YouTube API
        api_key = os.getenv("YOUTUBE_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="YouTube API key not configured")
        
        youtube = googleapiclient.discovery.build(
            "youtube", "v3", developerKey=api_key
        )
        
        # Get comments
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=max_results,
            textFormat="plainText",
            order="relevance"
        )
        response = request.execute()
        
        comments = []
        for item in response.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "text": snippet["textDisplay"],
                "author": snippet["authorDisplayName"],
                "published_at": snippet["publishedAt"],
                "likes": snippet["likeCount"]
            })
        
        return comments
    except googleapiclient.errors.HttpError as e:
        if "commentsDisabled" in str(e):
            raise HTTPException(status_code=400, detail="Comments are disabled for this video")
        else:
            raise HTTPException(status_code=500, detail=f"YouTube API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching comments: {str(e)}")

def analyze_sentiment_traditional(text: str) -> Dict[str, Any]:
    """Analyze sentiment using VADER only"""
    analyzer = SentimentIntensityAnalyzer()
    vader_scores = analyzer.polarity_scores(text)
    compound_score = vader_scores["compound"]
    
    # Determine sentiment label
    if compound_score >= 0.05:
        sentiment_label = "positive"
    elif compound_score <= -0.05:
        sentiment_label = "negative"
    else:
        sentiment_label = "neutral"
    
    return {
        "score": compound_score,
        "label": sentiment_label
    }

def analyze_sentiment_svm(text: str) -> Dict[str, Any]:
    """Analyze sentiment using SVM pipeline model"""
    global svm_model
    
    if svm_model is None:
        print(f"SVM model not available, falling back to traditional method")
        return analyze_sentiment_traditional(text)
    
    try:
        # Preprocess text
        preprocessed_text = preprocess_text(text)
        print(f"Preprocessed text sample: {preprocessed_text[:50]}...")
        
        # Get raw scores from SVM
        raw_scores = svm_model.decision_function([preprocessed_text])[0]
        print(f"Raw scores: {raw_scores}")
        
        # Get prediction (class with highest score)
        prediction = np.argmax(raw_scores)
        
        # Map prediction to label
        label_map = {0: "negative", 1: "neutral", 2: "positive"}
        sentiment_label = label_map.get(prediction, "neutral")
        
        # Calculate normalized score (-1 to 1 range)
        # Using softmax-like normalization
        exp_scores = np.exp(raw_scores - np.max(raw_scores))
        probabilities = exp_scores / exp_scores.sum()
        
        # Calculate final score based on probabilities
        if sentiment_label == "positive":
            score = probabilities[2]  # Probability of positive class
        elif sentiment_label == "negative":
            score = -probabilities[0]  # Negative probability of negative class
        else:
            score = 0  # Neutral
        
        print(f"Final sentiment: {sentiment_label}, score: {score}")
            
        return {
            "score": float(score),
            "label": sentiment_label
        }
    except Exception as e:
        print(f"Error in SVM sentiment analysis: {str(e)}")
        print(f"Stack trace: ", e)
        return analyze_sentiment_traditional(text)

@app.get("/")
async def root():
    return {"message": "YouTube Comment Sentiment Analysis API"}

@app.post("/upload-model")
async def upload_model(model_file: UploadFile = File(...)):
    """Upload a custom model file"""
    try:
        # Save model file
        model_path = os.path.join(MODEL_DIR, "best_model_svm_tfidf.pkl")
        with open(model_path, "wb") as f:
            f.write(await model_file.read())
        
        # Reload model
        global svm_model
        svm_model = joblib.load(model_path)
        
        # Verify it's a pipeline
        if not hasattr(svm_model, 'named_steps'):
            raise HTTPException(status_code=400, detail="Uploaded model is not a pipeline. Please upload a pipeline model.")
        
        return {"message": "Model uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading model: {str(e)}")

@app.post("/analyze", response_model=SentimentResponse)
async def analyze_youtube_comments(youtube_data: YouTubeURL):
    """
    Analyze sentiment of comments from a YouTube video
    """
    try:
        print(f"Request received with data: {youtube_data}")
        print(f"Using SVM model: {youtube_data.use_svm_model}")
        
        video_id = extract_video_id(youtube_data.url)
        video_info = get_video_info(video_id)
        comments_data = get_comments(video_id, youtube_data.max_comments)
        
        # Analyze sentiment for each comment
        analyzed_comments = []
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        # Determine which sentiment analysis method to use
        should_use_svm = youtube_data.use_svm_model and svm_model is not None
        model_used = "SVM with TF-IDF" if should_use_svm else "VADER"
        sentiment_analyzer = analyze_sentiment_svm if should_use_svm else analyze_sentiment_traditional
        
        print(f"Selected model: {model_used}")
        
        for comment in comments_data:
            sentiment = sentiment_analyzer(comment["text"])
            analyzed_comment = Comment(
                text=comment["text"],
                author=comment["author"],
                published_at=comment["published_at"],
                likes=comment["likes"],
                sentiment_score=sentiment["score"],
                sentiment_label=sentiment["label"]
            )
            
            analyzed_comments.append(analyzed_comment)
            
            # Update sentiment counts
            if sentiment["label"] == "positive":
                positive_count += 1
            elif sentiment["label"] == "negative":
                negative_count += 1
            else:  # neutral
                neutral_count += 1
        
        # Calculate percentages
        total_comments = len(analyzed_comments)
        summary = SentimentSummary(
            positive=round(positive_count / total_comments * 100, 2) if total_comments > 0 else 0,
            negative=round(negative_count / total_comments * 100, 2) if total_comments > 0 else 0,
            neutral=round(neutral_count / total_comments * 100, 2) if total_comments > 0 else 0
        )
        
        return SentimentResponse(
            comments=analyzed_comments,
            summary=summary,
            video_info=video_info,
            model_used=model_used
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing comments: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 