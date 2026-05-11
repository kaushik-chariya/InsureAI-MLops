# 🚗 Vehicle MLOps Project

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Database-336791?style=for-the-badge&logo=postgresql&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-Cloud-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-CI%2FCD-2088FF?style=for-the-badge&logo=githubactions&logoColor=white)

**An end-to-end Machine Learning pipeline with automated training, evaluation, and deployment using AWS, Docker, and GitHub Actions CI/CD.**

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [PostgreSQL Setup](#-postgresql-setup)
- [Pipeline Components](#-pipeline-components)
- [AWS Configuration](#-aws-configuration)
- [CI/CD Deployment](#-cicd-deployment)
- [Running the App](#-running-the-app)

---

## 🌟 Overview

This project implements a production-ready MLOps pipeline for vehicle data, covering the full lifecycle from **data ingestion** to **model deployment**. It integrates:

- 📦 Modular Python packaging with `setup.py` & `pyproject.toml`
- 🗄️ PostgreSQL as the backend data store
- ☁️ AWS S3 for model registry & ECR + EC2 for deployment
- 🔁 GitHub Actions for automated CI/CD

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10 |
| Database | PostgreSQL + SQLAlchemy + psycopg2 |
| Cloud Storage | AWS S3 |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Deployment | AWS EC2 + ECR |
| ML & Data | pandas, scikit-learn |
| Web App | FastAPI / Flask (`app.py`) |
| Environment | Conda virtual environment |

---

## 📁 Project Structure

```
vehicle-mlops/
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   │   └── model_evaluation.py
│   │   └── model_pusher.py
│   │
│   ├── configuration/
│   │   ├── postgres_db_connection.py
│   │   └── aws_connection.py
│   │
│   ├── entity/
│   │   ├── config_entity.py
│   │   ├── artifact_entity.py
│   │   ├── estimator.py
│   │   └── s3_estimator.py
│   │
│   ├── data_access/
│   │   └── proj1_data.py
│   │
│   ├── aws_storage/
│   │
│   ├── pipeline/
│   │   └── training_pipeline.py
│   │
│   ├── constants/
│   │   └── __init__.py
│   │
│   └── utils/
│       └── main_utils.py
│
├── config/
│   └── schema.yaml
│
├── notebook/
│   ├── EDA.ipynb
│   └── postgres_demo.ipynb
│
├── static/
├── templates/
│
├── .github/
│   └── workflows/
│       └── aws.yaml
│
├── app.py
├── demo.py
├── template.py
├── setup.py
├── pyproject.toml
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── .gitignore
```

---

## 🚀 Getting Started

### 1. Create Project Template

```bash
python template.py
```

### 2. Install Local Packages

Configure `setup.py` and `pyproject.toml` for local package imports, then:

```bash
# Create and activate Conda environment
conda create -n vehicle python=3.10 -y
conda activate vehicle

# Install all dependencies
pip install -r requirements.txt

# Verify local packages are installed
pip list
```

---

## 🗄️ PostgreSQL Setup

### Installation & Database Creation

1. Install PostgreSQL and ensure it is running.
2. Open PostgreSQL and create a new database:
   ```sql
   CREATE DATABASE vehicle_db;
   ```
3. Verify and connect to the database.

### Required Libraries (`requirements.txt`)

```
pandas
sqlalchemy
psycopg2-binary
ipykernel
```

### Connect via SQLAlchemy (Notebook)

```python
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("postgresql+psycopg2://<user>:<password>@localhost:5432/vehicle_db")
df = pd.read_csv("your_dataset.csv")
df.to_sql("your_table", con=engine, if_exists="replace", index=False)
```

### Set the PostgreSQL Connection URL

**Bash (Mac/Linux/Windows WSL):**
```bash
export POSTGRES_URL="postgresql+psycopg2://<username>:<password>@localhost:5432/vehicle_db"
echo $POSTGRES_URL
```

**PowerShell (Windows):**
```powershell
$env:POSTGRES_URL = "postgresql+psycopg2://<username>:<password>@localhost:5432/vehicle_db"
echo $env:POSTGRES_URL
```

> ⚠️ Add the `artifact/` directory to `.gitignore`.

---

## 🔧 Pipeline Components

Each component follows this standard workflow:

> `constants/__init__.py` → `config_entity.py` → `artifact_entity.py` → `component.py` → `training_pipeline.py` → `demo.py`

### ① Data Ingestion
- Connects to PostgreSQL via `postgres_db_connection.py`
- Fetches data using `pd.read_sql()` in `data_access/proj1_data.py`
- Outputs raw dataset artifact

### ② Data Validation
- Schema defined in `config/schema.yaml`
- Utility functions in `utils/main_utils.py`
- Validates column types, null values, and distributions

### ③ Data Transformation
- Preprocessing and feature engineering
- `estimator.py` added to `entity/` folder

### ④ Model Trainer
- Trains ML model on transformed data
- Model class added to `estimator.py`

### ⑤ Model Evaluation
- Compares new model against production model
- Threshold: **`MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE = 0.02`**

### ⑥ Model Pusher
- Pushes best model to **AWS S3** bucket: `my-model-mlopsproj`
- S3 Key: `model-registry`

---

## ☁️ AWS Configuration

### IAM Setup

1. Log into AWS Console (Region: **us-east-1**)
2. Go to **IAM → Create User** (e.g., `firstproj`)
3. Attach **AdministratorAccess** policy
4. Create **Access Key** (CLI type) → Download CSV

### Set AWS Environment Variables

**Bash:**
```bash
export AWS_ACCESS_KEY_ID="your_access_key"
export AWS_SECRET_ACCESS_KEY="your_secret_key"
```

**PowerShell:**
```powershell
$env:AWS_ACCESS_KEY_ID="your_access_key"
$env:AWS_SECRET_ACCESS_KEY="your_secret_key"
```

### S3 Bucket

- Go to **S3 → Create Bucket**
- Name: `my-model-mlopsproj` | Region: `us-east-1`
- Uncheck **"Block all public access"** → Acknowledge → Create

### Constants (`constants/__init__.py`)

```python
AWS_ACCESS_KEY_ID_ENV_KEY = "AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY_ENV_KEY = "AWS_SECRET_ACCESS_KEY"
REGION_NAME = "us-east-1"
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02
MODEL_BUCKET_NAME = "my-model-mlopsproj"
MODEL_PUSHER_S3_KEY = "model-registry"
```

---

## 🔁 CI/CD Deployment

### Docker Setup

```dockerfile
# Dockerfile + .dockerignore already configured
```

### GitHub Actions Workflow (`.github/workflows/aws.yaml`)

Triggered on every push to `main`.

### ECR Repository

```
AWS Console → ECR → Create Repository → Name: vehicleproj → Copy URI
```

### EC2 Instance

- **Name:** `vehicledata-machine`
- **Image:** Ubuntu Server 24.04 (free tier AMI)
- **Instance type:** T2 Medium
- **Storage:** 30 GB
- Allow HTTP & HTTPS traffic | Create new key pair: `proj1key`

### Install Docker on EC2

```bash
sudo apt-get update -y && sudo apt-get upgrade -y
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker
```

### Connect GitHub to EC2 (Self-Hosted Runner)

```
GitHub Repo → Settings → Actions → Runners → New Self-Hosted Runner → Linux
```
Run all **Download** commands on EC2, then configure:
```bash
# When prompted:
# Runner group  → press Enter
# Runner name   → self-hosted
# Extra label   → press Enter
# Work folder   → press Enter

./run.sh   # Starts the runner (state: idle in GitHub)
```

### GitHub Secrets

Go to: **GitHub Repo → Settings → Secrets and Variables → Actions → New Repository Secret**

| Secret Name | Value |
|---|---|
| `AWS_ACCESS_KEY_ID` | Your AWS Access Key |
| `AWS_SECRET_ACCESS_KEY` | Your AWS Secret Key |
| `AWS_DEFAULT_REGION` | `us-east-1` |
| `ECR_REPO` | Your ECR URI |

---

## 🌐 Running the App

### Expose EC2 Port

```
EC2 → Security → Security Groups → Edit Inbound Rules →
Add Rule: Custom TCP | Port: 8000 | Source: 0.0.0.0/0 → Save
```

### Access the Application

```
http://<your-ec2-public-ip>:8000
```

### Trigger Model Training

```
http://<your-ec2-public-ip>:8000/training
```

---

## 📝 Logging & Exception Handling

| File | Purpose |
|---|---|
| `logger.py` | Centralized logging setup; tested via `demo.py` |
| `exception.py` | Custom exception classes; tested via `demo.py` |

---

<div align="center">

Made with ❤️ for end-to-end MLOps

</div>

---

## 👨‍💻 Author

<div align="center">

### Kaushik Chariya

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Kaushik_Chariya-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kaushik-chariya/)
[![GitHub](https://img.shields.io/badge/GitHub-kaushik--chariya-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/kaushik-chariya)

*Building production-ready ML systems, one pipeline at a time.*

</div>