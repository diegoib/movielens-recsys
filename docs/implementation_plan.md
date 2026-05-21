# Plan de Implementación: movielens-recsys

Sistema de recomendación end-to-end sobre MovieLens 20M. Cada fase es una iteración entregable (un PR). Ver @docs/project_summary.md para el diseño completo del sistema.

---

## Índice

- [Stack tecnológico](#stack-tecnológico)
- [Estimación de costes GCP](#estimación-de-costes-gcp)
- [Fase 0 — Fundación del proyecto](#fase-0--fundación-del-proyecto)
- [Fase 1 — Infraestructura base (Terraform)](#fase-1--infraestructura-base-terraform)
- [Fase 2 — Pipeline de datos offline (Simulador 1)](#fase-2--pipeline-de-datos-offline-simulador-1)
- [Fase 3 — Feature Engineering (offline)](#fase-3--feature-engineering-offline)
- [Fase 4 — Modelo v1 (local + debug)](#fase-4--modelo-v1-local--debug)
- [Fase 5 — MLflow en GCP + Cloud Run Job de entrenamiento](#fase-5--mlflow-en-gcp--cloud-run-job-de-entrenamiento)
- [Fase 6 — Serving (FastAPI + ONNX)](#fase-6--serving-fastapi--onnx)
- [Fase 7 — Stack de streaming (RedPanda + PyFlink + Redis)](#fase-7--stack-de-streaming-redpanda--pyflink--redis)
- [Fase 8 — Simulador 2 (eventos en tiempo real)](#fase-8--simulador-2-eventos-en-tiempo-real)
- [Fase 9 — Reentrenamiento periódico (Airflow)](#fase-9--reentrenamiento-periódico-airflow)
- [Fase 10 — Monitoring y observabilidad](#fase-10--monitoring-y-observabilidad)
- [Fase 11 — CI/CD completo (GitHub Actions)](#fase-11--cicd-completo-github-actions)
- [Makefile — referencia completa](#makefile--referencia-completa)

---

## Stack tecnológico

| Componente | Local | GCP |
|---|---|---|
| Mensajería | RedPanda (Docker) | RedPanda en VM preemptible |
| Stream processing | PyFlink (Docker) | PyFlink en VM preemptible |
| Feature store | Redis (Docker) | Redis en VM preemptible |
| Model training | Lightning + uv local | Cloud Run Jobs |
| ML tracking | — (smoke test local, sin MLflow) | MLflow en VM (streaming-vm), acceso vía port forward |
| Serving | FastAPI + ONNX local | Cloud Run (scale-to-zero) |
| Orquestación | — | Airflow en e2-micro (free tier) |
| IaC | — | Terraform (state en GCS) |
| Monitoring | Prometheus + Grafana (Docker) | Prometheus + Grafana en VM |
| CI/CD | — | GitHub Actions + OIDC |
| Imágenes Docker | — | Artifact Registry |

---

## Estimación de costes GCP

Objetivo: ≤ 30 €/mes.

| Servicio | Configuración | Coste estimado |
|---|---|---|
| GCS | ~50 GB (data + modelos) | ~$1.50/mes |
| VM e2-medium preemptible | 24/7 (streaming stack) | ~$8/mes |
| VM e2-micro | Free tier (Airflow) | $0 |
| Cloud Run | Serving, scale-to-zero | ~$0.50/mes |
| Cloud Run Jobs | Training, ~2h/semana | ~$1/mes |
| Artifact Registry | ~5 GB imágenes | ~$0.50/mes |
| Cloud Monitoring | Free tier | $0 |
| **Total estimado** | | **~$11-12/mes** |

> La VM preemptible puede ser interrumpida por GCP en cualquier momento. El startup script la recupera automáticamente. Para un proyecto de aprendizaje, esto es aceptable.

---

## Estructura de directorios

```
movielens-recsys/
├── src/
│   ├── data/
│   │   ├── download.py           # descarga Kaggle
│   │   ├── generate_events.py    # Simulador 1
│   │   └── upload_gcs.py
│   ├── features/
│   │   ├── schema.py             # Pydantic schemas de features
│   │   ├── build_features.py     # feature engineering offline
│   │   └── processor.py          # PyFlink job (online)
│   ├── models/
│   │   ├── two_tower.py          # arquitectura PyTorch
│   │   ├── lightning_module.py   # Lightning wrapper + métricas
│   │   ├── export_onnx.py        # checkpoint → ONNX
│   │   └── promote.py            # staging → production en MLflow
│   ├── serving/
│   │   ├── app.py                # FastAPI
│   │   ├── scorer.py             # inferencia ONNX
│   │   └── candidates.py         # generación de candidatos
│   ├── simulator/
│   │   └── online_simulator.py   # Simulador 2
│   └── train.py                  # entrypoint de entrenamiento
├── infra/
│   ├── bootstrap/
│   │   └── create_state_bucket.sh
│   ├── modules/
│   │   ├── gcs/
│   │   ├── artifact-registry/
│   │   ├── iam/
│   │   ├── compute/
│   │   ├── cloud-run/
│   │   └── secrets/
│   └── environments/
│       ├── dev/
│       └── prod/
├── docker/
│   ├── training/Dockerfile
│   ├── serving/Dockerfile
│   └── streaming/Dockerfile      # PyFlink job
├── dags/
│   ├── weekly_retrain.py
│   └── daily_data_quality.py
├── docker-compose.yml            # stack local completo
├── .github/workflows/
│   ├── ci.yml
│   ├── deploy-serving.yml
│   └── deploy-training.yml
├── tests/
├── Makefile
└── docs/
    ├── project_summary.md
    └── implementation_plan.md    ← este archivo
```

---

## Fase 0 — Fundación del proyecto

**Objetivo**: estructura de proyecto lista, herramientas configuradas, GCP preparado para recibir Terraform.

**Definición de done**: `make lint` y `make test` pasan en CI; se puede hacer `make tf-init ENV=dev` sin error.

### Subtareas

**Proyecto GCP**
- [ ] Crear proyecto GCP (o reutilizar uno existente)
- [ ] Activar billing y configurar presupuesto con alerta en 25 €/mes
- [ ] Habilitar APIs: `cloudrun.googleapis.com`, `compute.googleapis.com`, `artifactregistry.googleapis.com`, `secretmanager.googleapis.com`, `cloudbuild.googleapis.com`, `iam.googleapis.com`, `storage.googleapis.com`

**Bootstrap de Terraform**
- [ ] Crear `infra/bootstrap/create_state_bucket.sh`: crea el bucket GCS para el state de Terraform (este paso es manual, no gestionado por TF para evitar el problema del huevo y la gallina)
- [ ] Ejecutar el script una sola vez: `bash infra/bootstrap/create_state_bucket.sh`

**Estructura del repositorio**
- [ ] Crear directorios vacíos con `.gitkeep`: `src/data/`, `src/features/`, `src/models/`, `src/serving/`, `src/simulator/`, `infra/modules/`, `infra/environments/dev/`, `infra/environments/prod/`, `docker/`, `dags/`, `tests/`
- [ ] Añadir `data/`, `mlruns/`, `.env` a `.gitignore`

**Makefile base**
- [ ] Target `help`: lista targets con descripción (usando comentarios `##`)
- [ ] Target `setup`: `uv sync --group all`
- [ ] Target `lint`: `uv run ruff check . && uv run mypy .`
- [ ] Target `fmt`: `uv run ruff format .`
- [ ] Target `test`: `uv run pytest`
- [ ] Variable `ENV` con default `dev` para comandos Terraform

**OIDC GitHub ↔ GCP**
- [ ] Crear Workload Identity Pool y Provider en GCP para GitHub Actions
- [ ] Service account `github-actions@<project>.iam.gserviceaccount.com` con roles mínimos (Artifact Registry Writer, Cloud Run Developer, etc.)
- [ ] Guardar `WORKLOAD_IDENTITY_PROVIDER` y `SERVICE_ACCOUNT` como GitHub Secrets del repositorio
- [ ] Verificar con un workflow mínimo que la autenticación funciona

```makefile
# Comandos de esta fase
make setup
make lint
make test
make tf-init ENV=dev
```

---

## Fase 1 — Infraestructura base (Terraform)

**Objetivo**: toda la infraestructura GCP declarada en código, aplicable en un solo `make tf-apply`.

**Definición de done**: `terraform apply` sin errores; los buckets, VM y service accounts existen en GCP.

### Módulos Terraform

**`modules/gcs/`**
- [ ] Bucket `data-raw-<project>`: datos crudos de MovieLens
- [ ] Bucket `data-processed-<project>`: eventos y features en Parquet
- [ ] Bucket `models-<project>`: checkpoints y archivos ONNX
- [ ] Bucket `mlflow-artifacts-<project>`: artifacts de MLflow
- [ ] Todos con versioning habilitado y lifecycle rule para borrar versiones >30 días

**`modules/artifact-registry/`**
- [ ] Repositorio Docker `movielens-recsys` en la región elegida

**`modules/iam/`**
- [ ] SA `cloud-run-serving@...`: roles `storage.objectViewer`, `secretmanager.secretAccessor`
- [ ] SA `cloud-run-jobs@...`: roles `storage.objectAdmin`, `artifactregistry.reader`
- [ ] SA `github-actions@...`: roles para push a Artifact Registry y deploy de Cloud Run

**`modules/compute/`**
- [ ] VM `streaming-vm`: e2-medium, preemptible, Ubuntu 22.04
  - Startup script: instala Docker + Docker Compose, clona repo, `docker compose up -d`
  - Service account con acceso a GCS para descargar config
- [ ] VM `airflow-vm`: e2-micro (free tier), Ubuntu 22.04
  - Startup script: instala Docker, arranca Airflow con LocalExecutor

**`modules/cloud-run/`**
- [ ] Servicio `mlflow-server`: placeholder (imagen pública de MLflow), conectado al bucket de artifacts
- [ ] Servicio `recsys-serving`: placeholder, se actualizará en Fase 6

**`modules/secrets/`**
- [ ] Secret `kaggle-username` y `kaggle-key`: vacíos, se rellenan manualmente después de apply

**Environments**
- [ ] `environments/dev/main.tf`: usa módulos con configuración reducida (VM más pequeña, un solo bucket)
- [ ] `environments/prod/main.tf`: configuración completa
- [ ] Variables comunes en `environments/variables.tf`

```makefile
make tf-init ENV=dev
make tf-plan ENV=dev
make tf-apply ENV=dev
```

---

## Fase 2 — Pipeline de datos offline (Simulador 1)

**Objetivo**: generar la tabla de eventos sintéticos (~150-170M filas) a partir de MovieLens 20M y subirla a GCS.

**Definición de done**: `data/processed/events.parquet` existe local y en GCS; los tests de validación pasan.

### Subtareas

**Descarga**
- [ ] `src/data/download.py`: usa la librería `kaggle` para descargar `grouplens/movielens-20m-dataset`
  - Lee `KAGGLE_USERNAME` y `KAGGLE_KEY` de variables de entorno (o de GCP Secrets Manager con `google-cloud-secret-manager`)
  - Output: `data/raw/ratings.csv`, `movies.csv`, `genome-scores.csv`, `genome-tags.csv`, `tags.csv`, `links.csv`

**Validación con Pydantic**
- [ ] `src/data/schemas.py`: modelos Pydantic para `Rating`, `Movie`, `GenomeScore`, y `Event`
- [ ] Validar que `ratings.csv` tiene las columnas esperadas y tipos correctos al cargar

**Generación de eventos (`src/data/generate_events.py`)**
- [ ] Paso 1 — Construir sesiones: agrupar ratings de cada usuario por proximidad temporal (gap > 60 min = nueva sesión); asignar `session_id` UUID
- [ ] Paso 2 — Funnel positivo: por cada rating real, generar hacia atrás: `impression` (t - 20-40 min), `view` (t - 15-25 min), `click` (t - 5-15 min), `rating` (t real de MovieLens)
- [ ] Paso 3 — Negativos tipo A (impresiones sin view): 4-6 por película puntuada en la sesión
  - ~40% películas populares en ventana de ±30 días
  - ~40% mismo género que las positivas de la sesión
  - ~20% películas que usuarios similares consumieron (collaborative filtering simple: Jaccard sobre usuarios)
  - Filtro: nunca usar película que el usuario haya puntuado alguna vez
- [ ] Paso 4 — Negativos tipo B (views sin click): 1 por cada 2-3 clicks en la sesión; películas similares al positivo pero con diferencia sutil
- [ ] Label binario: `click` y `rating` → label=1; `impression` y `view` sin click → label=0
- [ ] Output: `data/processed/events.parquet` con columnas: `event_id`, `timestamp`, `user_id`, `movie_id`, `event_type`, `rating` (nullable), `session_id`, `label`
- [ ] Usar Polars para el procesamiento (ya en deps) por performance con 150M+ filas

**Upload**
- [ ] `src/data/upload_gcs.py`: sube `data/raw/` y `data/processed/` a los buckets correspondientes

**Tests**
- [ ] `tests/test_generate_events.py`:
  - Ningún negativo de un usuario coincide con película que ese usuario haya puntuado
  - Todos los `event_type` válidos: `impression`, `view`, `click`, `rating`
  - Cada `click` tiene exactamente un `view` y una `impression` anteriores en la misma sesión
  - El ratio de negativos/positivos está en el rango esperado (4-6x)

```makefile
make data-download
make data-generate
make data-upload
```

---

## Fase 3 — Feature Engineering (offline)

**Objetivo**: materializar las features que usará el modelo en entrenamiento y serving.

**Definición de done**: `data/processed/train_dataset.parquet` existe con features de usuario y película; `make features` termina sin error.

### Subtareas

**Schemas**
- [ ] `src/features/schema.py`: dataclasses o Pydantic para `UserFeatures` y `MovieFeatures`

**Features de usuario** (`src/features/build_features.py`)
- [ ] `genre_affinity_last_7d`: vector de afinidad por género (calculado sobre clicks en los últimos 7 días de historial)
- [ ] `n_clicks_last_7d`: número de clicks en los últimos 7 días
- [ ] `avg_session_length`: media de películas por sesión
- [ ] `favorite_genres`: top-3 géneros por frecuencia de clicks históricos
- [ ] `days_since_last_activity`: días desde el último evento del usuario

**Features de película**
- [ ] `popularity_last_30d`: número de ratings/views en una ventana de 30 días alrededor del timestamp de entrenamiento
- [ ] `avg_rating`: media de ratings del MovieLens original
- [ ] `year`: extraído del título con regex `\((\d{4})\)`
- [ ] `genres_vector`: one-hot encoding de los 20 géneros del dataset
- [ ] `genome_top20`: top-20 genome scores por `relevance` para cada película

**Dataset de entrenamiento**
- [ ] `src/features/build_features.py`: join eventos + user features + movie features → `data/processed/train_dataset.parquet`
- [ ] Split temporal documentado: filas hasta el día X para train, día X+1 en adelante para test (X = percentil 80 de timestamps)

```makefile
make features
```

---

## Fase 4 — Modelo v1 (local + debug)

**Objetivo**: implementar la arquitectura two-tower y verificar que el código es correcto con datos dummy. El entrenamiento real ocurre siempre en GCP (Fase 5). Local = smoke test.

**Definición de done**: `make train-local-debug` pasa con 10.000 filas dummy sin error; el modelo ONNX produce el mismo output que el modelo PyTorch con esos datos.

> **Filosofía local**: no se corre el dataset completo ni se loguea en MLflow en local. El objetivo es verificar que los cambios al código no rompen nada antes de enviar a GCP. MLflow vive en la VM (ver Fase 5).

### Subtareas

**Arquitectura** (`src/models/two_tower.py`)
- [ ] `UserTower(nn.Module)`: embedding de `user_id` (dim configurable) + MLP sobre features de comportamiento; output = vector de dimensión D
- [ ] `MovieTower(nn.Module)`: embedding de `movie_id` + MLP sobre genre vector + year + popularity + genome; output = vector de dimensión D
- [ ] `TwoTowerModel(nn.Module)`: combina las dos torres; score = dot product de los vectores; `.encode_user()` y `.encode_movie()` como métodos separados (necesarios para serving)

**DataModule** (`src/data/dataset.py`)
- [ ] `RecSysDataModule(L.LightningDataModule)`: carga `train_dataset.parquet` con Polars; split temporal; DataLoader con batches de pares (user_features, movie_features, label)
- [ ] Parámetro `max_rows: int | None = None` para limitar a N filas en smoke test local
- [ ] Normalización de features continuas (mean/std calculado solo sobre train split)

**Lightning Module** (`src/models/lightning_module.py`)
- [ ] `TwoTowerLightningModule(L.LightningModule)`: BCE loss; optimizador Adam; log de `train_loss`, `val_loss`, `val_auc`, `val_ndcg5`, `val_precision5`
- [ ] Métricas con `torchmetrics` (ya incluido en `lightning[pytorch-extra]`)
- [ ] `MLFLOW_TRACKING_URI` leído de variable de entorno; si no está definida, omitir logging MLflow (permite correr en local sin servidor MLflow)

**Entrypoint** (`src/train.py`)
- [ ] `tyro.cli(TrainConfig)` para hiperparámetros: `lr`, `batch_size`, `embedding_dim`, `mlp_hidden_dims`, `max_epochs`, `max_rows`, `fast_dev_run`
- [ ] Cuando `MLFLOW_TRACKING_URI` no está definida: entrenar sin tracking (solo para smoke test local)
- [ ] Cuando está definida (en GCP): loguear hiperparámetros, métricas por época y artefactos

**Export ONNX** (`src/models/export_onnx.py`)
- [ ] Cargar checkpoint Lightning; exportar con `torch.onnx.export()`; verificar con `onnxruntime.InferenceSession` que el output coincide con PyTorch (tolerancia 1e-5)
- [ ] Output: `artifacts/models/model_v<N>.onnx`

**Smoke test local**
- [ ] `make train-local-debug`: corre `src/train.py --max_rows 10000 --max_epochs 2 --fast_dev_run True` con datos reales pero limitados; sin MLflow
- [ ] Verifica que el pipeline completo (carga → forward pass → loss → backward → export ONNX) funciona sin errores

```makefile
make train-local-debug    # smoke test: 10K filas, 2 épocas, sin MLflow
```

---

## Fase 5 — MLflow en VM + Cloud Run Job de entrenamiento

**Objetivo**: el entrenamiento ocurre en la nube, los modelos quedan registrados en MLflow con versionado. MLflow corre en la `streaming-vm` junto al resto del stack; se accede via port forward.

**Definición de done**: `make train-gcp` lanza un Cloud Run Job que termina con éxito y registra el modelo en MLflow con stage `Staging`; la UI de MLflow es accesible en `localhost:5000` via port forward.

### Subtareas

**MLflow en la VM (añadir a `docker-compose.yml`)**
- [ ] Servicio `mlflow-server`: imagen `ghcr.io/mlflow/mlflow`; puerto 5000; artifact store apuntando al bucket GCS `mlflow-artifacts-<project>`
- [ ] Backend store: SQLite en un volumen Docker persistente (sobrevive reinicios del contenedor; el volumen vive en el disco de la VM)
- [ ] Acceso: `gcloud compute ssh streaming-vm -- -L 5000:localhost:5000` para abrir la UI en el browser local
- [ ] El puerto 5000 ya está configurado en el devcontainer para autoforwarding

**Dockerfile de training** (`docker/training/Dockerfile`)
- [ ] Base: `python:3.12-slim`
- [ ] Instalar `uv`, copiar `pyproject.toml`, ejecutar `uv sync --group train --group data`
- [ ] `ENTRYPOINT ["uv", "run", "python", "src/train.py"]`

**Adaptaciones de `src/train.py` para GCP**
- [ ] Leer `train_dataset.parquet` desde GCS usando `fsspec` + `gcsfs` (ambos ya disponibles via Polars/PyArrow)
- [ ] `MLFLOW_TRACKING_URI` desde variable de entorno (Cloud Run URL en producción, `mlruns/` en local)
- [ ] Guardar checkpoint y ONNX en GCS al finalizar

**Cloud Run Job (Terraform)**
- [ ] `modules/cloud-run-jobs/`: job `training-job`; 4 vCPU, 16 GB RAM; timeout 6h; service account `cloud-run-jobs@...`
- [ ] Variables de entorno inyectadas desde Secret Manager

**Registro y promoción**
- [ ] `src/models/promote.py`: compara `val_auc` del run actual vs el modelo en stage `Production`; llama `mlflow.MlflowClient().transition_model_version_stage()` si mejora

**Test local de la imagen**
- [ ] `docker build -f docker/training/Dockerfile -t recsys-training .`
- [ ] `docker run --env-file .env recsys-training` (con credenciales GCP montadas)

```makefile
make docker-build-training
make train-gcp
make model-promote
```

---

## Fase 6 — Serving (FastAPI + ONNX)

**Objetivo**: API de recomendaciones funcionando local y en Cloud Run, con latencia P95 < 200ms.

**Definición de done**: `curl http://localhost:8000/recommendations/123` devuelve 5 películas; deploy en Cloud Run pasa el smoke test.

### Subtareas

**Generación de candidatos** (`src/serving/candidates.py`)
- [ ] `generate_candidates(user_id, user_features, n=200)`: top-N por `popularity_last_30d` filtradas por géneros afines del usuario; excluir películas ya vistas
- [ ] Cargar índice de popularidad en memoria al arrancar el servidor

**Scorer ONNX** (`src/serving/scorer.py`)
- [ ] `OnnxScorer`: carga el modelo desde GCS al iniciar (path configurable via env var `MODEL_URI`)
- [ ] `score(user_features, movie_features_batch) -> List[float]`: batch inference con ONNXRuntime
- [ ] Cache de `MovieFeatures` en memoria (se cargan desde GCS al arrancar; ~27K películas caben fácilmente)

**FastAPI app** (`src/serving/app.py`)
- [ ] `GET /recommendations/{user_id}`:
  1. Leer `UserFeatures` desde Redis (key: `user:{user_id}`)
  2. Generar 200 candidatos
  3. Score con ONNX
  4. Devolver top-5 ordenados por score
- [ ] `POST /events`: valida el evento con Pydantic; publica en RedPanda (local) o topic Pub/Sub (GCP)
- [ ] `GET /health`: devuelve `{"status": "ok", "model_version": "..."}`
- [ ] `GET /metrics`: formato Prometheus (con `prometheus-fastapi-instrumentator`)
- [ ] Lifespan handler para cargar modelo, features y conectar Redis al arrancar

**Dockerfile** (`docker/serving/Dockerfile`)
- [ ] Base: `python:3.12-slim`; `uv sync --group onnx`; `ENTRYPOINT ["uv", "run", "uvicorn", "src.serving.app:app"]`

**Cloud Run (Terraform)**
- [ ] Actualizar módulo `cloud-run/` para `recsys-serving`: scale-to-zero, max 3 instancias, concurrencia 80, 1 vCPU / 512 MB
- [ ] Variable de entorno `REDIS_HOST` apuntando a la IP interna de la VM de streaming

**Tests**
- [ ] Test de integración local con Docker Compose: verificar que `/recommendations/{user_id}` devuelve 5 películas con campos `movie_id`, `title`, `score`
- [ ] `make serve-local`: arranca FastAPI + Redis local para debugging manual

```makefile
make serve-local      # FastAPI en localhost:8000 con Redis local
make serve-deploy     # build + push + deploy en Cloud Run
```

---

## Fase 7 — Stack de streaming (RedPanda + PyFlink + Redis)

**Objetivo**: pipeline en tiempo real funcionando: evento → RedPanda → PyFlink → features actualizadas en Redis.

**Definición de done**: emitir un evento de click hace que las features del usuario en Redis se actualicen en < 2 segundos.

### Subtareas

**Docker Compose local** (`docker-compose.yml`)
- [ ] Servicio `redpanda`: imagen oficial `redpandadata/redpanda`; topic `events` y topic `model-predictions` creados en `command`
- [ ] Servicio `redpanda-console`: UI web en puerto 8080 para inspeccionar mensajes
- [ ] Servicio `flink-jobmanager`: imagen `flink:1.18-python3`; expone puerto 8081 (Flink UI)
- [ ] Servicio `flink-taskmanager`: misma imagen, conectado al jobmanager
- [ ] Servicio `redis`: imagen `redis:7-alpine`; puerto 6379
- [ ] Servicio `recsys-serving`: imagen local, depende de redis

**PyFlink job** (`src/features/processor.py`)
- [ ] Conectar a RedPanda via Kafka connector (RedPanda es compatible con la API de Kafka)
- [ ] `StreamExecutionEnvironment` con source en topic `events`
- [ ] Operador de ventana deslizante: por `user_id`, ventana de 1h actualizada cada 1 min
- [ ] Calcular: `n_clicks_last_1h`, `genre_affinity_last_1h` (actualización incremental)
- [ ] Sink: escritura en Redis con `redis-py` en un operador custom; key `user:{user_id}`, TTL 2h

**`docker/streaming/Dockerfile`**
- [ ] Imagen para el job PyFlink: base `flink:1.18-python3`; instala `apache-flink`, `redis`, `kafka-python`

**VM de streaming en GCP**
- [ ] Startup script en Terraform (`modules/compute/`): instala Docker + Docker Compose; descarga `docker-compose.yml` desde GCS; ejecuta `docker compose up -d`
- [ ] Script de recuperación ante preemption: systemd service que ejecuta `docker compose up -d` al arrancar
- [ ] Regla de firewall para que Cloud Run pueda acceder a Redis (puerto 6379) vía IP interna
- [ ] Abrir puerto 9092 para RedPanda (solo desde IPs internas de GCP)

**Verificación del pipeline completo**
- [ ] `make streaming-local`: arranca todo el stack; enviar un evento de click manualmente a RedPanda; verificar en Redis que las features del usuario se actualizaron

```makefile
make streaming-local     # docker compose up stack completo
make streaming-deploy    # provisionamiento + startup script en VM GCP
make streaming-status    # ssh a VM y ver estado de los contenedores
```

---

## Fase 8 — Simulador 2 (eventos en tiempo real)

**Objetivo**: feedback loop completo. El modelo influye en los datos que recibe; las features del usuario cambian durante la sesión.

**Definición de done**: con N=10 usuarios simulados, el endpoint `/recommendations` devuelve resultados diferentes para el mismo usuario antes y después de que éste "haga click" en una película de terror.

### Subtareas

**`src/simulator/online_simulator.py`**
- [ ] Clase `User`: estado interno (historial de clicks en sesión actual, géneros vistos)
- [ ] Clase `SimulatorConfig` (tyro): `n_users`, `events_per_second`, `api_url`, `redpanda_brokers`
- [ ] Por cada usuario concurrente (asyncio o threading):
  1. Login → llama `GET /recommendations/{user_id}`
  2. Para cada película recomendada: decide si hacer click con probabilidad `sigmoid(score * temperatura + ruido)`
  3. Emite evento a RedPanda: `POST /events` o directamente a Kafka topic
  4. Si hay click: espera 30-120s simulando "ver la película", luego vuelve al paso 1
  5. Logout después de N páginas o tiempo máximo de sesión
- [ ] Log de inferencia: emitir al topic `model-predictions` los campos `user_id`, `movie_id`, `score`, `position`, `model_version`, `timestamp`
- [ ] Configurable: `--temperatura` controla cuánto sigue el modelo las recomendaciones (1.0 = sigue bien, 0.1 = casi aleatorio)

**Verificación del feedback loop**
- [ ] Script de smoke test: emitir 5 clicks de terror para `user_id=1`; esperar 2s; verificar que `/recommendations/1` ahora tiene más películas de terror en el top-5

**Deploy**
- [ ] El simulador puede correr localmente (`make simulate N=10`) o como Cloud Run Job en GCP (`make simulate-gcp N=1000`)

```makefile
make simulate N=10          # simulación local con 10 usuarios
make simulate-gcp N=1000    # Cloud Run Job con 1000 usuarios
```

---

## Fase 9 — Reentrenamiento periódico (Airflow)

**Objetivo**: el modelo se reentrena automáticamente cada semana con datos nuevos; solo se promueve si mejora AUC.

**Definición de done**: el DAG `weekly_retrain` se ejecuta en Airflow, el Cloud Run Job de training se lanza y el modelo se promueve automáticamente si mejora.

### Subtareas

**Dataset de reentrenamiento** (`src/data/build_retrain_dataset.py`)
- [ ] Leer log de inferencia del topic `model-predictions` (persistido en GCS por PyFlink o por sink del simulador)
- [ ] Join con eventos reales: si el usuario clickó en una película recomendada → label=1; si no → label=0
- [ ] Output: `data/processed/retrain_dataset_<fecha>.parquet`

**Airflow en VM e2-micro**
- [ ] `docker-compose.yml` en la VM Airflow: servicios `airflow-webserver` + `airflow-scheduler` con LocalExecutor y backend SQLite (simple, suficiente para 1-2 DAGs)
- [ ] Imagen base `apache/airflow:2.9.2-python3.12`; instalar `apache-airflow-providers-google` para interactuar con GCP

**DAG `weekly_retrain`** (`dags/weekly_retrain.py`)
```
1. PythonOperator: ejecutar build_retrain_dataset.py (lee de GCS)
2. PythonOperator: feature engineering sobre nuevos datos
3. GoogleCloudRunJobOperator: lanzar Cloud Run Job de training
4. CloudRunJobSensor: esperar a que el job termine (polling cada 5 min)
5. PythonOperator: ejecutar promote.py (promueve si AUC mejora)
6. SlackWebhookOperator (opcional): notificar resultado
```
- [ ] Trigger: `@weekly`, empezando el lunes a las 02:00 UTC
- [ ] En caso de fallo: reintento automático una vez; alerta por email si falla dos veces

**DAG `daily_data_quality`** (`dags/daily_data_quality.py`)
- [ ] Verificar que el número de eventos del día anterior supera un mínimo
- [ ] Verificar distribución de `event_type` no ha derivado > 20%
- [ ] Alerta si alguna check falla

**Deploy Airflow**
- [ ] `make airflow-deploy`: sube los DAGs a la VM via SCP o git pull desde la VM; reinicia el scheduler

```makefile
make airflow-deploy    # deploy DAGs a VM e2-micro
make retrain-manual    # trigger manual del DAG weekly_retrain
```

---

## Fase 10 — Monitoring y observabilidad

**Objetivo**: métricas de negocio y técnicas visibles en Grafana; alertas configuradas.

**Definición de done**: el dashboard de Grafana muestra CTR rolling en tiempo real mientras el simulador emite eventos.

### Subtareas

**Métricas en FastAPI** (añadir a `src/serving/app.py`)
- [ ] Contador `recommendations_served_total` por `user_id` (opcional) y `model_version`
- [ ] Contador `clicks_total` y `impressions_total` (vía `POST /events`)
- [ ] Gauge `ctr_1h`: ratio clicks/impresiones en ventana deslizante de 1h (calculado en background task)
- [ ] Gauge `ctr_24h`: ídem para 24h
- [ ] Gauge `unique_movies_recommended_24h`: set de películas recomendadas en 24h
- [ ] Histograma `inference_latency_seconds` con buckets P50/P95/P99
- [ ] Gauge `score_distribution_mean` y `score_distribution_std`
- [ ] Gauge `redis_feature_age_seconds`: tiempo desde la última actualización de features en Redis (detecta feature staleness)

**Prometheus** (añadir al `docker-compose.yml` local y a la VM de streaming)
- [ ] Imagen `prom/prometheus:latest`; config para scraping de FastAPI `/metrics` cada 15s
- [ ] Alertas en `alert_rules.yml`:
  - CTR cae > 20% en 1h respecto a la hora anterior
  - `inference_latency_seconds` P95 > 0.5s
  - `redis_feature_age_seconds` > 300s (features no se actualizan en 5 min)

**Grafana** (añadir al `docker-compose.yml` local y a la VM de streaming)
- [ ] Imagen `grafana/grafana-oss:latest`; datasource Prometheus configurado via provisioning
- [ ] Dashboard de negocio: CTR 1h y 24h, películas únicas recomendadas, rating post-click
- [ ] Dashboard técnico: latencia P50/P95, score distribution, feature freshness
- [ ] Dashboard de modelo: AUC histórico por versión de modelo (datos desde MLflow API)

```makefile
make monitoring-local    # arranca Prometheus + Grafana en Docker Compose local
```

---

## Fase 11 — CI/CD completo (GitHub Actions)

**Objetivo**: cada PR valida calidad; cada push a main despliega automáticamente.

**Definición de done**: abrir una PR activa el workflow de CI; el merge a main despliega el serving a Cloud Run sin intervención manual.

### Subtareas

**`.github/workflows/ci.yml`** (trigger: PR a cualquier rama)
- [ ] Job `lint`: `uv run ruff check . && uv run mypy .`
- [ ] Job `test`: `uv run pytest --cov=src --cov-report=xml`; publicar coverage en el PR
- [ ] Job `docker-build`: `docker build` para cada Dockerfile (sin push); verifica que las imágenes compilan

**`.github/workflows/deploy-serving.yml`** (trigger: push a `main`)
- [ ] Autenticación OIDC con `google-github-actions/auth`
- [ ] `docker build + push` de la imagen serving a Artifact Registry
- [ ] `gcloud run deploy recsys-serving --image <nueva-imagen> --region <region>`
- [ ] Smoke test: `curl https://<cloud-run-url>/health` y verificar `{"status": "ok"}`

**`.github/workflows/deploy-training.yml`** (trigger: `workflow_dispatch` manual)
- [ ] Autenticación OIDC
- [ ] `docker build + push` de la imagen training
- [ ] `gcloud run jobs update training-job --image <nueva-imagen>`

**Secrets de GitHub requeridos**
- [ ] `WORKLOAD_IDENTITY_PROVIDER`: creado en Fase 0
- [ ] `SERVICE_ACCOUNT`: email del SA de GitHub Actions
- [ ] `GCP_PROJECT_ID`: ID del proyecto GCP

```makefile
# CI/CD se activa automáticamente; no hay targets de Makefile específicos
# Para forzar deploy manual:
make serve-deploy
```

---

## Makefile — referencia completa

| Target | Descripción |
|---|---|
| `make help` | Lista todos los targets con descripción |
| `make setup` | `uv sync --group all` |
| `make lint` | `ruff check` + `mypy` |
| `make fmt` | `ruff format` |
| `make test` | `pytest` |
| `make data-download` | Descarga MovieLens 20M desde Kaggle |
| `make data-generate` | Ejecuta Simulador 1 (genera eventos históricos) |
| `make data-upload` | Sube raw + processed a GCS |
| `make features` | Feature engineering offline |
| `make train-local` | Entrena modelo localmente |
| `make train-local-debug` | `--fast_dev_run True` (1 batch) |
| `make docker-build-training` | Build imagen Docker de training |
| `make train-gcp` | Lanza Cloud Run Job de training |
| `make model-promote` | Promueve modelo si mejora AUC |
| `make serve-local` | FastAPI + Redis en localhost:8000 |
| `make serve-deploy` | Build + push + deploy en Cloud Run |
| `make streaming-local` | Docker Compose: stack completo local |
| `make streaming-deploy` | Provisionamiento + startup en VM GCP |
| `make streaming-status` | SSH a VM y ver estado de contenedores |
| `make simulate N=10` | Simulador 2 local con N usuarios |
| `make simulate-gcp N=1000` | Simulador 2 como Cloud Run Job |
| `make monitoring-local` | Arranca Prometheus + Grafana local |
| `make airflow-deploy` | Despliega DAGs a VM e2-micro |
| `make retrain-manual` | Trigger manual del DAG weekly_retrain |
| `make tf-init ENV=dev` | `terraform init` en el environment indicado |
| `make tf-plan ENV=dev` | `terraform plan` |
| `make tf-apply ENV=dev` | `terraform apply` |
| `make docker-build` | Build todas las imágenes Docker |
