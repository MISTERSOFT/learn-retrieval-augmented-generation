#!/bin/bash

# Download RAG movies dataset
curl -o data/movies.json https://storage.googleapis.com/qvault-webapp-dynamic-assets/course_assets/course-rag-movies.json

echo "Download complete: data/movies.json"