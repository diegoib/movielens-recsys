# Runbook — movielens-recsys

Guía paso a paso para ejecutar el proyecto desde cero. Cubre hasta la Fase 2 (pipeline de datos offline). Cada sección indica claramente qué necesitas tener instalado y qué comandos ejecutar.

---

## Requisitos previos

Herramientas que deben estar instaladas antes de empezar:

| Herramienta | Versión mínima | Verificar con |
|---|---|---|
| [uv](https://docs.astral.sh/uv/getting-started/installation/) | 0.4+ | `uv --version` |
| [gcloud CLI](https://cloud.google.com/sdk/docs/install) | cualquiera | `gcloud --version` |
| [Terraform](https://developer.hashicorp.com/terraform/install) | 1.5+ | `terraform --version` |
| Git | cualquiera | `git --version` |

---

## Fase 0 — Preparar el entorno local

### 1. Clonar el repositorio e instalar dependencias

```bash
git clone https://github.com/<tu-usuario>/movielens-recsys.git
cd movielens-recsys
uv sync --group all
```

Verifica que el entorno está bien:

```bash
make lint    # debe terminar sin errores
make test    # debe pasar todos los tests
```

### 2. Crear y configurar el proyecto GCP

Crea el proyecto en la [consola de GCP](https://console.cloud.google.com/) o con:

```bash
gcloud projects create <GCP_PROJECT_ID> --name="movielens-recsys"
gcloud config set project <GCP_PROJECT_ID>
```

Activa el billing para el proyecto (obligatorio para usar Compute y Cloud Run). Configura también una alerta de presupuesto en 25 €/mes desde **Billing → Budgets & alerts**.

### 3. Habilitar las APIs de GCP

```bash
export GCP_PROJECT_ID=<tu-project-id>
bash infra/bootstrap/enable_apis.sh
```

El script habilita: Cloud Run, Compute Engine, Artifact Registry, Secret Manager, Cloud Build, IAM, GCS, IAM Credentials, Cloud Scheduler, Firestore y Pub/Sub.

### 4. Configurar OIDC para GitHub Actions

Esto permite que GitHub Actions se autentique en GCP sin almacenar claves de servicio.

```bash
export GCP_PROJECT_ID=<tu-project-id>
export GITHUB_REPO=<usuario>/<repo>   # ej: diegoib/movielens-recsys
bash infra/bootstrap/setup_oidc.sh
```

El script imprime al final tres valores. Añádelos como **GitHub Secrets** en `Settings → Secrets and variables → Actions` de tu repositorio:

| Secret | Valor |
|---|---|
| `WORKLOAD_IDENTITY_PROVIDER` | El que imprime el script |
| `SERVICE_ACCOUNT` | El que imprime el script |
| `GCP_PROJECT_ID` | Tu project ID |

---

## Fase 1 — Infraestructura en GCP (Terraform)

### 5. Inicializar Terraform

```bash
make tf-init
```

### 6. Revisar el plan de infraestructura

```bash
make tf-plan GCP_PROJECT_ID=<tu-project-id>
```

Revisa el output. Terraform creará:
- Bucket GCS `<project-id>-data` (datos, modelos, artefactos de MLflow)
- Repositorio Docker `movielens-recsys` en Artifact Registry
- Service accounts: `cloud-run-serving`, `cloud-run-jobs`
- VM `streaming-vm` (e2-medium, preemptible) — RedPanda + PyFlink + Redis + MLflow
- VM `airflow-vm` (e2-micro, free tier) — Airflow
- Servicio Cloud Run `recsys-serving` (placeholder)

### 7. Aplicar la infraestructura

```bash
make tf-apply GCP_PROJECT_ID=<tu-project-id>
```

Escribe `yes` cuando Terraform lo solicite.

**Verificar que los recursos existen:**

```bash
# Bucket creado
gcloud storage buckets describe gs://<tu-project-id>-data

# VMs creadas
gcloud compute instances list --project=<tu-project-id>

# Repositorio de imágenes
gcloud artifacts repositories list --project=<tu-project-id>
```

> **Nota sobre la VM preemptible**: GCP puede interrumpir `streaming-vm` en cualquier momento. El startup script la recupera automáticamente al reiniciar. Para un proyecto de aprendizaje esto es aceptable.

---

## Fase 2 — Pipeline de datos offline (Simulador 1)

### 8. Descargar MovieLens

```bash
make data-download
```

Descarga `ml-latest.zip` (~250 MB) desde [grouplens.org](https://files.grouplens.org/datasets/movielens/ml-latest.zip) y extrae los CSV en `data/raw/`. La descarga es idempotente: si el zip ya existe, se omite.

Archivos resultantes en `data/raw/`:

```
ratings.csv          # ~20M filas: userId, movieId, rating, timestamp
movies.csv           # ~62K películas: movieId, title, genres
genome-scores.csv    # relevancia de tags por película
genome-tags.csv      # nombres de los tags del genome
tags.csv             # tags libres de usuarios
links.csv            # IDs de IMDB y TMDB
```

### 9. Generar la tabla de eventos sintéticos

El Simulador 1 transforma los 20M de ratings en ~150-170M de eventos. Requiere al menos 16 GB de RAM libre. Elige una de las dos rutas:

#### Ruta A — Local (solo si tienes ≥ 16 GB RAM libre)

```bash
make data-generate
```

#### Ruta B — GCP con datagen-vm (recomendada)

La VM `datagen-vm` (e2-highmem-4, 32 GB RAM) se crea apagada. La levantas solo para este paso y la vuelves a apagar al terminar.

**1. Aplicar infraestructura** (si aún no lo has hecho en el paso 7):

```bash
make tf-apply GCP_PROJECT_ID=<tu-project-id>
```

**2. Arrancar la VM:**

```bash
make datagen-vm-start GCP_PROJECT_ID=<tu-project-id>
```

Espera ~60 s hasta que la VM esté en estado `RUNNING`:

```bash
gcloud compute instances describe datagen-vm \
    --zone us-central1-a --project <tu-project-id> \
    --format="value(status)"
```

**3. (Solo la primera vez) Clonar el repositorio dentro de la VM:**

```bash
gcloud compute ssh datagen-vm --project <tu-project-id> --zone us-central1-a
# dentro de la VM:
git clone https://github.com/<tu-usuario>/movielens-recsys.git
exit
```

**4. Lanzar el pipeline desacoplado de la terminal:**

```bash
make datagen-run GCP_PROJECT_ID=<tu-project-id>
```

El comando termina inmediatamente en tu terminal local. El pipeline corre dentro de una sesión `tmux` en la VM — sobrevive a desconexiones SSH. Al terminar (incluyendo la subida a GCS), **la VM se apaga sola**.

**5. Monitorizar el progreso (desde cualquier terminal, en cualquier momento):**

```bash
# Comprobación rápida sin conectarse
make datagen-status GCP_PROJECT_ID=<tu-project-id>

# Engancharse y ver la barra tqdm en vivo
make datagen-attach GCP_PROJECT_ID=<tu-project-id>
# Para desengancharse sin matar el proceso: Ctrl+B  D
```

> La VM en estado `TERMINATED` en la consola GCP es la señal definitiva de que el pipeline ha terminado correctamente (la propia VM se apaga al acabar). No hace falta `datagen-vm-stop`.

---

Qué hace `data-generate` internamente:
1. Agrupa los ratings de cada usuario en sesiones (gap > 60 min = nueva sesión)
2. Por cada rating real genera el funnel positivo hacia atrás: `impression → view → click → rating`
3. Añade 4-6 impresiones negativas por película puntuada (40% populares, 40% mismo género, 20% colaborativo)
4. Añade 1 view-sin-click por cada 2-3 clicks (negativos difíciles)
5. Escribe `data/processed/events.parquet`

**Verificar el output** (en local o dentro de la VM antes de subir):

```bash
uv run python - <<'EOF'
import polars as pl
df = pl.read_parquet("data/processed/events.parquet")
print(f"Filas totales : {df.height:,}")
print(f"event_types   : {df['event_type'].value_counts().sort('count', descending=True)}")
print(f"Columnas      : {df.columns}")
EOF
```

Valores esperados:
- Filas: entre 150M y 170M
- Los cuatro `event_type` presentes: `impression`, `view`, `click`, `rating`

**Ejecutar los tests de validación** (en local):

```bash
make test
```

Los tests comprueban que ningún negativo coincide con una película que el usuario haya puntuado, que el funnel click→view→impression es coherente, y que el ratio negativos/positivos está en el rango esperado.

### 10. Subir los datos a GCS

> Si usaste la **Ruta B**, los datos ya están en GCS — puedes saltar este paso.

Si usaste la **Ruta A** (generación local):

```bash
GCP_PROJECT_ID=<tu-project-id> make data-upload
```

Sube `data/raw/*.csv` y `data/processed/events.parquet` al bucket `<project-id>-data`.

**Verificar la subida:**

```bash
gcloud storage ls gs://<tu-project-id>-data/raw/
gcloud storage ls gs://<tu-project-id>-data/processed/
```

---

## Resumen de comandos

```bash
# Entorno local
uv sync --group all
make lint && make test

# GCP bootstrap (una sola vez)
export GCP_PROJECT_ID=<id>
export GITHUB_REPO=<usuario>/<repo>
bash infra/bootstrap/enable_apis.sh
bash infra/bootstrap/setup_oidc.sh

# Infraestructura
make tf-init
make tf-plan  GCP_PROJECT_ID=<id>
make tf-apply GCP_PROJECT_ID=<id>

# Pipeline de datos (Ruta B — GCP)
make datagen-vm-start GCP_PROJECT_ID=<id>

# (Solo la primera vez) clonar el repo dentro de la VM
gcloud compute ssh datagen-vm --project <id> --zone us-central1-a
git clone https://github.com/<usuario>/movielens-recsys.git && exit

# Lanzar el pipeline desacoplado (tmux, la VM se apaga sola al terminar)
make datagen-run GCP_PROJECT_ID=<id>

# Monitorizar
make datagen-status GCP_PROJECT_ID=<id>   # comprobación rápida
make datagen-attach GCP_PROJECT_ID=<id>   # ver tqdm en vivo (Ctrl+B D para desengancharse)
```
