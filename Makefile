GCP_PROJECT_ID ?= movielens-recsys-proj
GCP_REGION ?= us-central1
GCP_ZONE ?= us-central1-a

.PHONY: help setup lint fmt test \
        tf-init tf-plan tf-apply \
        data-download data-generate data-upload features \
        train-local train-local-debug train-gcp model-promote \
        docker-build docker-build-training \
        serve-local serve-deploy \
        streaming-local streaming-deploy streaming-status \
        simulate simulate-gcp \
        airflow-deploy retrain-manual \
        monitoring-local \
        datagen-vm-start datagen-vm-stop \
        datagen-run datagen-attach datagen-status

help: ## List all available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-28s\033[0m %s\n", $$1, $$2}'

# ── Dev environment ───────────────────────────────────────────────────────────

setup: ## Install all dependency groups
	uv sync --group all

lint: ## Run ruff check + mypy
	uv run ruff check .
	uv run mypy .

fmt: ## Format code with ruff
	uv run ruff format .

test: ## Run test suite
	uv run pytest --tb=short

# ── Terraform ─────────────────────────────────────────────────────────────────

tf-init: ## Initialize Terraform
	cd infra && terraform init

tf-plan: ## Preview Terraform changes
	cd infra && terraform plan \
		-var="project_id=$(GCP_PROJECT_ID)" \
		-var="region=$(GCP_REGION)"

tf-apply: ## Apply Terraform changes
	cd infra && terraform apply \
		-var="project_id=$(GCP_PROJECT_ID)" \
		-var="region=$(GCP_REGION)"

# ── Data pipeline ─────────────────────────────────────────────────────────────

data-download: ## Download MovieLens 20M from Kaggle
	uv run python src/data/download.py

data-generate: ## Run Simulator 1: generate historical events table
	uv run python src/data/generate_events.py

data-upload: ## Upload raw + processed data to GCS
	uv run python src/data/upload_gcs.py

features: ## Compute offline features from events
	uv run python src/features/build_features.py

# ── Model training ────────────────────────────────────────────────────────────

train-local-debug: ## Smoke test: 10K rows, 2 epochs, no MLflow
	uv run python src/train.py --max_rows 10000 --max_epochs 2 --fast_dev_run True

train-local: ## Full local training run (requires processed data)
	uv run python src/train.py

train-gcp: ## Trigger Cloud Run Job for training
	gcloud run jobs execute training-job --region $(GCP_REGION) --project $(GCP_PROJECT_ID)

model-promote: ## Promote model to production in MLflow if AUC improves
	uv run python src/models/promote.py

# ── Docker ────────────────────────────────────────────────────────────────────

docker-build: ## Build all Docker images
	docker build -f docker/training/Dockerfile  -t recsys-training .
	docker build -f docker/serving/Dockerfile   -t recsys-serving .
	docker build -f docker/streaming/Dockerfile -t recsys-streaming .

docker-build-training: ## Build training Docker image
	docker build -f docker/training/Dockerfile -t recsys-training .

# ── Serving ───────────────────────────────────────────────────────────────────

serve-local: ## Run FastAPI + Redis locally on localhost:8000
	docker compose up recsys-serving redis

serve-deploy: ## Build, push and deploy serving image to Cloud Run
	docker build -f docker/serving/Dockerfile \
		-t $(GCP_REGION)-docker.pkg.dev/$(GCP_PROJECT_ID)/movielens-recsys/serving:latest .
	docker push $(GCP_REGION)-docker.pkg.dev/$(GCP_PROJECT_ID)/movielens-recsys/serving:latest
	gcloud run deploy recsys-serving \
		--image $(GCP_REGION)-docker.pkg.dev/$(GCP_PROJECT_ID)/movielens-recsys/serving:latest \
		--region $(GCP_REGION) --project $(GCP_PROJECT_ID)

# ── Streaming stack ───────────────────────────────────────────────────────────

streaming-local: ## Start full local stack via Docker Compose
	docker compose up

streaming-deploy: ## Deploy streaming stack to GCP preemptible VM
	gcloud compute ssh streaming-vm --project $(GCP_PROJECT_ID) -- \
		"cd /opt/movielens-recsys && git pull && docker compose up -d"

streaming-status: ## Check container status on streaming VM
	gcloud compute ssh streaming-vm --project $(GCP_PROJECT_ID) -- "docker compose ps"

# ── Simulator 2 ───────────────────────────────────────────────────────────────

simulate: ## Run Simulator 2 locally (N=10 default users)
	uv run python src/simulator/online_simulator.py --n_users $(or $(N),10)

simulate-gcp: ## Run Simulator 2 as Cloud Run Job (N=1000 default)
	gcloud run jobs execute simulator-job \
		--region $(GCP_REGION) --project $(GCP_PROJECT_ID) \
		--args="--n_users,$(or $(N),1000)"

# ── Airflow ───────────────────────────────────────────────────────────────────

airflow-deploy: ## Deploy DAGs to Airflow VM
	gcloud compute scp dags/ airflow-vm:/opt/airflow/dags --recurse \
		--project $(GCP_PROJECT_ID)
	gcloud compute ssh airflow-vm --project $(GCP_PROJECT_ID) -- \
		"docker compose restart airflow-scheduler"

retrain-manual: ## Manually trigger the weekly_retrain DAG
	gcloud compute ssh airflow-vm --project $(GCP_PROJECT_ID) -- \
		"docker compose exec airflow-scheduler airflow dags trigger weekly_retrain"

# ── Monitoring ────────────────────────────────────────────────────────────────

monitoring-local: ## Start Prometheus + Grafana locally
	docker compose up prometheus grafana

# ── Datagen VM ────────────────────────────────────────────────────────────────

datagen-vm-start: ## Arrancar datagen-vm en GCP
	gcloud compute instances start datagen-vm \
		--zone $(GCP_ZONE) --project $(GCP_PROJECT_ID)

datagen-vm-stop: ## Apagar datagen-vm en GCP (después de generar los datos)
	gcloud compute instances stop datagen-vm \
		--zone $(GCP_ZONE) --project $(GCP_PROJECT_ID)

datagen-run: ## Lanzar pipeline completo en tmux (survives SSH disconnect; VM se apaga sola al terminar)
	gcloud compute ssh datagen-vm --project $(GCP_PROJECT_ID) --zone $(GCP_ZONE) -- \
		"tmux new-session -d -s datagen 'cd ~/movielens-recsys && uv sync --group data && make data-download && make data-generate && GCP_PROJECT_ID=$(GCP_PROJECT_ID) make data-upload && sudo shutdown -h now'"

datagen-attach: ## Reengancharse al tmux session para ver el progreso en vivo (desengancharse: Ctrl+B D)
	gcloud compute ssh datagen-vm --project $(GCP_PROJECT_ID) --zone $(GCP_ZONE) \
		--ssh-flag="-t" -- "tmux attach -t datagen"

datagen-status: ## Comprobar si el pipeline sigue corriendo en datagen-vm
	gcloud compute ssh datagen-vm --project $(GCP_PROJECT_ID) --zone $(GCP_ZONE) -- \
		"tmux ls 2>/dev/null && echo 'Pipeline en progreso' || echo 'Sin sesión activa (terminado o no iniciado)'"
