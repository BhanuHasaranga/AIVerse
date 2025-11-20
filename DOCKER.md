# 🐳 Docker Setup for AI/ML Learning Platform

## Prerequisites

- Docker installed ([Download Docker](https://www.docker.com/products/docker-desktop))
- Docker Compose (comes with Docker Desktop)

## Quick Start

### 1. Build and Run with Docker Compose (Recommended)

```bash
# Build the image and start the container
docker-compose up --build

# The app will be available at: http://localhost:8501
```

### 2. Build Manually (Optional)

```bash
# Build the Docker image
docker build -t aiml-learning:latest .

# Run the container
docker run -p 8501:8501 -v $(pwd):/app aiml-learning:latest

# On Windows PowerShell:
docker run -p 8501:8501 -v ${PWD}:/app aiml-learning:latest
```

## Commands

### Start the Application
```bash
docker-compose up
```

### Start in Background
```bash
docker-compose up -d
```

### Stop the Application
```bash
docker-compose down
```

### View Logs
```bash
docker-compose logs -f
```

### Rebuild After Changes
```bash
docker-compose up --build
```

### Remove Everything (Including Images)
```bash
docker-compose down --rmi all
```

## Access the Application

Once running, open your browser and navigate to:
```
http://localhost:8501
```

## Project Structure in Container

```
/app
├── main.py                 # Entry point
├── pages/                  # Multi-page modules
├── utils/                  # Utility functions
├── requirements.txt        # Python dependencies
├── Dockerfile             # Container definition
└── docker-compose.yml     # Compose configuration
```

## Environment Variables

The following environment variables are set in the container:

- `STREAMLIT_SERVER_PORT=8501` — Port to run on
- `STREAMLIT_SERVER_ADDRESS=0.0.0.0` — Listen on all interfaces
- `STREAMLIT_SERVER_HEADLESS=true` — Run in headless mode (no browser auto-open)

## Performance Tips

1. **Hot Reload**: Changes to Python files are reflected immediately (volume mounted)
2. **Caching**: Streamlit caches computations automatically
3. **Memory**: Docker container has access to host system resources

## Troubleshooting

### Port Already in Use
```bash
# Change port in docker-compose.yml
# ports:
#   - "8502:8501"  # Use 8502 instead

docker-compose up
# Then access at http://localhost:8502
```

### Permission Denied (Linux/Mac)
```bash
# Run with proper permissions
sudo docker-compose up
```

### Out of Memory
```bash
# Increase Docker memory limit in Docker Desktop settings
# Or use memory limits in docker-compose.yml:
# services:
#   aiml-learning:
#     mem_limit: 2g
```

### Container Won't Start
```bash
# Check logs
docker-compose logs

# Rebuild from scratch
docker-compose down --rmi all
docker-compose up --build
```

## Production Deployment

For production, consider:

1. **Use a reverse proxy** (Nginx, Traefik)
2. **Add authentication** (Streamlit Cloud, custom middleware)
3. **Enable HTTPS** (SSL certificates)
4. **Use environment files** (.env)
5. **Set resource limits** in docker-compose.yml
6. **Use named volumes** for persistent data

Example production docker-compose.yml:

```yaml
version: '3.8'
services:
  aiml-learning:
    build: .
    container_name: aiml-learning-prod
    ports:
      - "127.0.0.1:8501:8501"
    mem_limit: 1g
    cpus: '1.5'
    restart: always
    environment:
      - STREAMLIT_SERVER_HEADLESS=true
      - STREAMLIT_LOGGER_LEVEL=warning
```

## Clean Up

```bash
# Remove stopped containers
docker container prune

# Remove unused images
docker image prune

# Remove everything (be careful!)
docker system prune -a
```

## Support

For more information:
- [Docker Documentation](https://docs.docker.com/)
- [Streamlit Docker Guide](https://docs.streamlit.io/knowledge-base/tutorials/deploy/docker)
- [Docker Compose Reference](https://docs.docker.com/compose/compose-file/)
