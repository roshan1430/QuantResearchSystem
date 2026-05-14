# Deployment Guide

## One-command startup

```bash
docker-compose up --build
```

## Exposed services
- Frontend: `http://localhost:5173`
- Backend OpenAPI: `http://localhost:8000/docs`
- MLflow: `http://localhost:5000`
- PostgreSQL: `localhost:5432`
- Kafka host listener: `localhost:29092`

## Next hardening steps
- add Alembic migrations
- add secrets management
- pin model artifacts in object storage
- switch frontend container to an optimized production build
