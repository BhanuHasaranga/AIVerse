@echo off
REM Quick start script for AI/ML Learning Platform with Docker

echo.
echo ======================================
echo  AI/ML Learning Platform - Docker
echo ======================================
echo.

REM Check if Docker is installed
docker --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker is not installed or not in PATH
    echo Please install Docker Desktop from: https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)

REM Check if docker-compose is available
docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: docker-compose not found
    echo It should come with Docker Desktop
    pause
    exit /b 1
)

echo Docker found: 
docker --version
echo.

REM Menu
:menu
echo.
echo Choose an option:
echo 1. Start application (docker-compose up)
echo 2. Start in background (docker-compose up -d)
echo 3. Stop application (docker-compose down)
echo 4. Rebuild and start (docker-compose up --build)
echo 5. View logs (docker-compose logs -f)
echo 6. Clean up everything
echo 7. Exit
echo.

set /p choice="Enter choice (1-7): "

if "%choice%"=="1" (
    echo.
    echo Starting application...
    docker-compose up
    goto menu
)

if "%choice%"=="2" (
    echo.
    echo Starting application in background...
    docker-compose up -d
    echo.
    echo Application started! Open: http://localhost:8501
    echo.
    goto menu
)

if "%choice%"=="3" (
    echo.
    echo Stopping application...
    docker-compose down
    echo Application stopped.
    echo.
    goto menu
)

if "%choice%"=="4" (
    echo.
    echo Rebuilding and starting...
    docker-compose up --build
    goto menu
)

if "%choice%"=="5" (
    echo.
    echo Showing logs (Ctrl+C to exit)...
    docker-compose logs -f
    goto menu
)

if "%choice%"=="6" (
    echo.
    echo WARNING: This will remove all containers and images
    set /p confirm="Are you sure? (yes/no): "
    if /i "%confirm%"=="yes" (
        echo Cleaning up...
        docker-compose down --rmi all
        echo Cleanup complete.
    )
    echo.
    goto menu
)

if "%choice%"=="7" (
    echo Goodbye!
    exit /b 0
)

echo Invalid choice. Please try again.
goto menu
