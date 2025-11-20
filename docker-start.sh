#!/bin/bash

# Quick start script for AI/ML Learning Platform with Docker

echo ""
echo "======================================"
echo " AI/ML Learning Platform - Docker"
echo "======================================"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed"
    echo "Please install Docker from: https://www.docker.com/products/docker-desktop"
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "ERROR: docker-compose not found"
    echo "It should come with Docker Desktop"
    exit 1
fi

echo "Docker version:"
docker --version
echo ""

# Menu loop
while true; do
    echo ""
    echo "Choose an option:"
    echo "1. Start application (docker-compose up)"
    echo "2. Start in background (docker-compose up -d)"
    echo "3. Stop application (docker-compose down)"
    echo "4. Rebuild and start (docker-compose up --build)"
    echo "5. View logs (docker-compose logs -f)"
    echo "6. Clean up everything"
    echo "7. Exit"
    echo ""
    
    read -p "Enter choice (1-7): " choice
    
    case $choice in
        1)
            echo ""
            echo "Starting application..."
            docker-compose up
            ;;
        2)
            echo ""
            echo "Starting application in background..."
            docker-compose up -d
            echo ""
            echo "Application started! Open: http://localhost:8501"
            echo ""
            ;;
        3)
            echo ""
            echo "Stopping application..."
            docker-compose down
            echo "Application stopped."
            echo ""
            ;;
        4)
            echo ""
            echo "Rebuilding and starting..."
            docker-compose up --build
            ;;
        5)
            echo ""
            echo "Showing logs (Ctrl+C to exit)..."
            docker-compose logs -f
            ;;
        6)
            echo ""
            echo "WARNING: This will remove all containers and images"
            read -p "Are you sure? (yes/no): " confirm
            if [ "$confirm" = "yes" ]; then
                echo "Cleaning up..."
                docker-compose down --rmi all
                echo "Cleanup complete."
            fi
            echo ""
            ;;
        7)
            echo "Goodbye!"
            exit 0
            ;;
        *)
            echo "Invalid choice. Please try again."
            ;;
    esac
done
