# Bootstrap

Scripts de configuración one-time para el proyecto en GCP. Ejecutar en orden.

## Prerequisitos

- `gcloud` CLI instalado y autenticado (`gcloud auth login`)

## Orden de ejecución

```bash
# 1. Crea el proyecto GCP (si no existe aún)
gcloud projects create movielens-recsys-proj --name="MovieLens RecSys"
# Después: activa billing en console.cloud.google.com → Billing → Link account
# Sin billing activo, enable_apis.sh fallará aunque seas el propietario.

# 2. Configura las variables de entorno (puedes añadirlas a tu .env)
export GCP_PROJECT_ID=movielens-recsys-proj
export GCP_REGION=us-central1
export GITHUB_REPO=diegoib/movielens-recsys

# 3. Habilita las APIs necesarias
bash infra/bootstrap/enable_apis.sh

# 4. Configura OIDC para GitHub Actions (sin service account keys)
bash infra/bootstrap/setup_oidc.sh
# → Copia los valores impresos como GitHub Secrets en tu repositorio

# 5. Inicializa Terraform (estado local)
make tf-init
```

## GitHub Secrets necesarios

| Secret | Descripción |
|---|---|
| `WORKLOAD_IDENTITY_PROVIDER` | Impreso por `setup_oidc.sh` |
| `SERVICE_ACCOUNT` | Impreso por `setup_oidc.sh` |
| `GCP_PROJECT_ID` | ID de tu proyecto GCP |

Todos los scripts son **idempotentes**: si el recurso ya existe, lo omiten.

## Estado de Terraform

El estado de Terraform se guarda localmente en `infra/terraform.tfstate`.
Este archivo está gitignoreado — no lo subas al repositorio.
