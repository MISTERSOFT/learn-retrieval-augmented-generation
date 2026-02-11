#!/bin/bash

# Download RAG movies dataset
curl -o data/movies.json https://storage.googleapis.com/qvault-webapp-dynamic-assets/course_assets/course-rag-movies.json
echo "Download complete: data/movies.json"

# Download stopwords dataset
curl -o data/stopwords.txt https://gist.githubusercontent.com/MISTERSOFT/6feda7be48a9d86124f2d2c8c1ea9998/raw/75001d5cef31a778dccf2763fb80158fe1ffdd9b/rag-test-stopwords.txt
echo "Download complete: data/stopwords.txt"