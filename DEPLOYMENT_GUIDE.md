# Deployment Guide

## Local Development

### Quick Start (Windows)
```bash
start.bat
```

### Quick Start (Linux/Mac)
```bash
bash start.sh
```

---

## Docker Deployment (Recommended)

### Prerequisites
- Docker Desktop installed
- 4GB+ available RAM
- 2GB+ disk space

### Build and Run
```bash
# Build images
docker-compose build

# Run services
docker-compose up

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f api
docker-compose logs -f frontend

# Stop services
docker-compose down

# Remove all volumes
docker-compose down -v
```

### Access Points
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:5000
- **Health Check**: http://localhost:5000/health

---

## Cloud Deployment

### AWS EC2 Deployment

1. **Launch EC2 Instance**
   ```bash
   # AMI: Ubuntu 22.04 LTS
   # Instance Type: t3.medium or larger
   # Storage: 30GB minimum
   ```

2. **Connect and Setup**
   ```bash
   ssh -i your-key.pem ubuntu@your-instance-ip
   
   # Update system
   sudo apt update && sudo apt upgrade -y
   
   # Install Docker
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   
   # Install Docker Compose
   sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
   sudo chmod +x /usr/local/bin/docker-compose
   ```

3. **Clone and Deploy**
   ```bash
   cd /home/ubuntu
   git clone <your-repo-url> lung-cancer-ai
   cd lung-cancer-ai
   
   # Create environment file
   cp .env.example .env
   # Edit .env with your settings
   
   # Start services
   sudo docker-compose up -d
   ```

4. **Configure Security Group**
   - Allow port 80 (HTTP)
   - Allow port 443 (HTTPS)
   - Allow port 3000 (Frontend)
   - Allow port 5000 (API)

5. **Setup SSL with Let's Encrypt**
   ```bash
   sudo apt install certbot python3-certbot-nginx -y
   sudo certbot certonly --standalone -d your-domain.com
   ```

### Google Cloud Platform (GCP)

1. **Using Cloud Run**
   ```bash
   # Build image
   gcloud builds submit --tag gcr.io/your-project/lung-cancer-api
   
   # Deploy
   gcloud run deploy lung-cancer-api \
     --image gcr.io/your-project/lung-cancer-api \
     --memory 4Gi \
     --region us-central1 \
     --allow-unauthenticated
   ```

2. **Using App Engine**
   ```yaml
   # app.yaml
   runtime: python39
   entrypoint: gunicorn -b :$PORT app:app
   
   env: standard
   
   env_variables:
     FLASK_ENV: "production"
   ```
   
   ```bash
   gcloud app deploy
   ```

### Azure App Service

1. **Create Resource Group**
   ```bash
   az group create --name lung-cancer-rg --location eastus
   ```

2. **Create App Service Plan**
   ```bash
   az appservice plan create \
     --name lung-cancer-plan \
     --resource-group lung-cancer-rg \
     --sku B2 \
     --is-linux
   ```

3. **Deploy**
   ```bash
   az webapp create \
     --resource-group lung-cancer-rg \
     --plan lung-cancer-plan \
     --name lung-cancer-app \
     --deployment-container-image-name-user-managed
   ```

---

## Production Considerations

### 1. Environment Variables
```env
# .env (production)
FLASK_ENV=production
DEBUG=False
FLASK_DEBUG=0
CORS_ALLOWED_ORIGINS=https://yourdomain.com
API_PORT=5000
WORKERS=4
```

### 2. WSGI Server (Production)
Replace Flask dev server with Gunicorn:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### 3. Reverse Proxy (Nginx)
```nginx
server {
    listen 80;
    server_name yourdomain.com;

    # Redirect to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    # Static files
    location /static {
        alias /app/frontend/build/static;
        expires 1y;
    }

    # API
    location /api/ {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # Frontend
    location / {
        proxy_pass http://localhost:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 4. Logging
```python
# In app.py
import logging
from logging.handlers import RotatingFileHandler

if not app.debug:
    file_handler = RotatingFileHandler('logs/flask.log', maxBytes=10240, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
```

### 5. Database (Optional)
Add PostgreSQL for user management and image history:

```yaml
services:
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_PASSWORD: yourpassword
    volumes:
      - postgres_data:/var/lib/postgresql/data
    
  api:
    depends_on:
      - postgres
    environment:
      DATABASE_URL: postgresql://user:password@postgres:5432/lung_cancer
```

### 6. Monitoring
```bash
# Docker stats
docker stats lung-cancer-api lung-cancer-ui

# Check logs
docker-compose logs --tail=100 -f api
```

### 7. Backup
```bash
# Backup models and data
tar -czf backup-$(date +%Y%m%d).tar.gz models/ data/meta/

# Upload to cloud storage
aws s3 cp backup-*.tar.gz s3://your-bucket/backups/
```

---

## SSL/TLS Configuration

### Using Let's Encrypt with Certbot
```bash
sudo certbot certonly \
  --standalone \
  -d yourdomain.com \
  -d www.yourdomain.com

# Update Nginx config with cert paths
sudo certbot renew --dry-run
```

### Using Self-Signed Certificate (Testing Only)
```bash
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes
```

---

## Monitoring and Health Checks

### Health Check Endpoint
```bash
# Set up regular health checks
curl http://localhost:5000/health

# Add to docker-compose.yml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
  interval: 30s
  timeout: 10s
  retries: 3
```

### Application Monitoring
```bash
# Option 1: Prometheus + Grafana
# Option 2: New Relic
# Option 3: DataDog
# Option 4: CloudWatch (AWS)
```

---

## Load Balancing

### Multiple Instances
```yaml
version: '3.8'

services:
  api-1:
    build: .
    ports:
      - "5001:5000"
  
  api-2:
    build: .
    ports:
      - "5002:5000"
  
  nginx:
    image: nginx:latest
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
```

### Nginx Load Balancing
```nginx
upstream api {
    server api-1:5000;
    server api-2:5000;
}

location /api/ {
    proxy_pass http://api;
}
```

---

## Performance Optimization

### Image Caching
```python
from flask_caching import Cache

cache = Cache(app, config={'CACHE_TYPE': 'simple'})

@app.route('/model-info')
@cache.cached(timeout=3600)
def model_info():
    # ...
```

### Database Connection Pooling
```python
from sqlalchemy import create_engine

engine = create_engine(
    'postgresql://...',
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=40
)
```

### CDN Integration
```html
<!-- Serve frontend assets from CDN -->
<link rel="stylesheet" href="https://cdn.example.com/app.css">
```

---

## Disaster Recovery

### Backup Strategy
```bash
# Daily automated backups
0 2 * * * tar -czf /backups/lung-cancer-$(date +\%Y\%m\%d).tar.gz /app/models /app/data
0 3 * * * aws s3 sync /backups s3://your-bucket/backups --delete
```

### Recovery Procedure
```bash
# Restore from backup
tar -xzf backup-20240115.tar.gz -C /app/

# Verify data integrity
python scripts/verify_models.py

# Restart services
docker-compose restart
```

---

## Troubleshooting

### Check Service Status
```bash
docker-compose ps

# Output should show:
# NAME                STATUS
# lung-cancer-api     Up
# lung-cancer-ui      Up
```

### View Logs
```bash
# All services
docker-compose logs

# Specific service
docker-compose logs api
docker-compose logs frontend

# Real-time
docker-compose logs -f
```

### Test API
```bash
# Health check
curl http://localhost:5000/health

# Predict
curl -X POST http://localhost:5000/predict \
  -F "image=@test.jpg"
```

---

## Cost Optimization

### Recommended Instance Types
| Provider | Type | Cost/Month | Specs |
|----------|------|-----------|-------|
| AWS | t3.medium | ~$30 | 2vCPU, 4GB RAM |
| GCP | n1-standard-2 | ~$50 | 2vCPU, 7.5GB RAM |
| Azure | B2s | ~$35 | 2vCPU, 4GB RAM |

### Cost Reduction
1. Use reserved instances (30-70% discount)
2. Auto-scaling based on demand
3. Use serverless for batch processing
4. Cache predictions (Redis)

---

## Compliance

### HIPAA Compliance (Healthcare)
- Encrypt data in transit (HTTPS)
- Encrypt data at rest
- Access controls and audit logs
- Regular backups
- Business Associate Agreement (BAA)

### GDPR Compliance
- Data retention policy
- User consent management
- Right to deletion
- Data portability

---

**Ready for Production!** 🚀

For issues, check logs and refer to [SETUP_GUIDE.md](SETUP_GUIDE.md)
