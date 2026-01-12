# CryptoBot Platform - Docker Setup

## Prerequisites

- Docker Desktop (Windows/Mac) or Docker Engine (Linux)
- At least 4GB RAM available for Docker

## Quick Start

### 1. Copy environment file

```bash
cp .env.example .env
```

Edit `.env` with your settings (database password, API keys, etc.)

### 2. Start all services

```bash
docker-compose up -d
```

### 3. Check status

```bash
docker-compose ps
```

You should see:
- `cryptobot-db` - Running (TimescaleDB)
- `cryptobot-app` - Running (Python app)
- `cryptobot-grafana` - Running (Dashboards)

### 4. View logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f cryptobot
docker-compose logs -f timescaledb
```

## Access Points

| Service | URL | Credentials |
|---------|-----|-------------|
| TimescaleDB | `localhost:5432` | See `.env` |
| Grafana | `http://localhost:3000` | admin / (see `.env`) |

## Common Commands

```bash
# Stop all services
docker-compose down

# Stop and remove all data (fresh start)
docker-compose down -v

# Rebuild after code changes
docker-compose up -d --build

# Enter Python container shell
docker exec -it cryptobot-app bash

# Enter database shell
docker exec -it cryptobot-db psql -U cryptobot

# Run a specific script
docker exec cryptobot-app python -m cryptobot.scripts.backtest
```

## Directory Structure

```
D:\CryptoBot\
├── docker-compose.yaml     # Service definitions
├── Dockerfile              # Python app build
├── requirements.txt        # Python dependencies
├── .env                    # Your secrets (not in git)
├── .env.example            # Template
│
├── cryptobot/              # Python source code
├── config/                 # YAML configuration
├── scripts/                # SQL and utility scripts
│
├── data/
│   ├── postgres/           # Database files (auto-created)
│   ├── grafana/            # Grafana data (auto-created)
│   └── csv/                # Your CSV data files
│
├── models/                 # Saved ML models
└── logs/                   # Application logs
```

## Portability

This entire setup runs from your portable drive. To move to another machine:

1. Install Docker on the new machine
2. Copy the entire `D:\CryptoBot` folder
3. Run `docker-compose up -d`

Database data persists in `data/postgres/`.

## First Time Database Setup

The database schema is automatically created on first run via `scripts/init_db.sql`.

To verify:

```bash
docker exec -it cryptobot-db psql -U cryptobot -c "\dt"
```

You should see tables: `ohlcv`, `features`, `signals`, `trades`, `portfolio`, `equity`, `models`, `backtest_results`

## Troubleshooting

### Port already in use

Change ports in `docker-compose.yaml`:
```yaml
ports:
  - "5433:5432"  # Use 5433 instead
```

### Database won't start

Check logs:
```bash
docker-compose logs timescaledb
```

Remove old data and restart:
```bash
docker-compose down -v
rm -rf data/postgres/*
docker-compose up -d
```

### Python app crashes

Check logs:
```bash
docker-compose logs cryptobot
```

Enter container to debug:
```bash
docker exec -it cryptobot-app bash
python -m cryptobot.main
```
