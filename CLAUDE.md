# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python/FastAPI application that generates hand-drawn style maps using AI. Users select a geographic area on a web map, and the system generates a stylized map using Stable Diffusion and ControlNet models.

## Key Components

1. **Frontend**: `index.html` - A Leaflet.js map interface that allows users to draw areas and displays generated maps
2. **Backend**: `main.py` - FastAPI server that handles API requests, GIS data fetching, and AI image generation
3. **AI Pipeline**: Uses Stable Diffusion v1.5 with ControlNet for image generation
4. **GIS Data**: Fetches OpenStreetMap data via Overpass API

## Development Commands

### Environment Setup
```bash
# Install uv if not already installed
pip install uv

# Create virtual environment
uv venv

# Activate virtual environment (macOS/Linux)
source .venv/bin/activate

# Activate virtual environment (Windows)
.venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
```

### Running the Application
```bash
# Start the FastAPI server
uvicorn main:app --reload
```

### Dependencies
- FastAPI for backend API
- Stable Diffusion and ControlNet for AI image generation
- OpenCV, PIL for image processing
- Requests for API calls
- Leaflet.js for frontend mapping

## Code Architecture

### Backend Structure (main.py)
- FastAPI application with endpoints for area selection and job status
- Background task processing for AI generation
- OSM data fetching via Overpass API
- Semantic map creation from GIS data
- AI image generation using ControlNet pipeline

### Frontend Structure (index.html)
- Leaflet.js map with drawing controls
- Job status polling and image overlay display
- UI controls for overlay visibility and opacity

### Data Flow
1. User draws area on map
2. Coordinates sent to backend `/api/start_generation`
3. Backend creates job and starts background task
4. Background task fetches OSM data, creates semantic map
5. AI model generates hand-drawn style image
6. Frontend polls `/api/job_status/{job_id}` until complete
7. Generated image returned as base64 and displayed as overlay

## Important Notes

- Large model files are cached in `./model_cache` directory
- Debug outputs are saved in `./debug_runs` with job-specific subdirectories
- First run will download several GB of AI models
- AI generation can take 2-5 minutes depending on hardware
- Supports CUDA, MPS (Mac), and CPU acceleration