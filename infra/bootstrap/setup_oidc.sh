#!/usr/bin/env bash
# Sets up Workload Identity Federation for GitHub Actions → GCP authentication.
# Eliminates the need to store service account keys as GitHub Secrets.
# Run once after enable_apis.sh.
set -euo pipefail

: "${GCP_PROJECT_ID:?Set GCP_PROJECT_ID before running this script}"
: "${GITHUB_REPO:?Set GITHUB_REPO as owner/repo (e.g. diegoib/movielens-recsys)}"
GCP_REGION="${GCP_REGION:-us-central1}"

PROJECT_NUMBER=$(gcloud projects describe "${GCP_PROJECT_ID}" --format="value(projectNumber)")
POOL_ID="github-pool"
PROVIDER_ID="github-provider"
SA_NAME="github-actions"
SA_EMAIL="${SA_NAME}@${GCP_PROJECT_ID}.iam.gserviceaccount.com"

echo "Project: ${GCP_PROJECT_ID} (${PROJECT_NUMBER})"
echo "Repo   : ${GITHUB_REPO}"

# 1. Workload Identity Pool
if gcloud iam workload-identity-pools describe "${POOL_ID}" \
    --location=global --project="${GCP_PROJECT_ID}" &>/dev/null; then
  echo "Pool '${POOL_ID}' already exists — skipping."
else
  gcloud iam workload-identity-pools create "${POOL_ID}" \
    --location=global \
    --display-name="GitHub Actions Pool" \
    --project="${GCP_PROJECT_ID}"
fi

# 2. OIDC Provider
if gcloud iam workload-identity-pools providers describe "${PROVIDER_ID}" \
    --workload-identity-pool="${POOL_ID}" \
    --location=global --project="${GCP_PROJECT_ID}" &>/dev/null; then
  echo "Provider '${PROVIDER_ID}' already exists — skipping."
else
  gcloud iam workload-identity-pools providers create-oidc "${PROVIDER_ID}" \
    --workload-identity-pool="${POOL_ID}" \
    --location=global \
    --issuer-uri="https://token.actions.githubusercontent.com" \
    --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
    --attribute-condition="attribute.repository=='${GITHUB_REPO}'" \
    --project="${GCP_PROJECT_ID}"
fi

# 3. Service account
if gcloud iam service-accounts describe "${SA_EMAIL}" --project="${GCP_PROJECT_ID}" &>/dev/null; then
  echo "SA '${SA_EMAIL}' already exists — skipping."
else
  gcloud iam service-accounts create "${SA_NAME}" \
    --display-name="GitHub Actions SA" \
    --project="${GCP_PROJECT_ID}"
fi

# 4. IAM roles for the SA
ROLES=(
  roles/artifactregistry.writer
  roles/run.developer
  roles/run.invoker
  roles/iam.serviceAccountUser
  roles/storage.objectViewer
)
for ROLE in "${ROLES[@]}"; do
  gcloud projects add-iam-policy-binding "${GCP_PROJECT_ID}" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="${ROLE}" \
    --condition=None \
    --quiet
done

# 5. Allow the Workload Identity Pool to impersonate the SA
POOL_RESOURCE="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL_ID}/attribute.repository/${GITHUB_REPO}"
gcloud iam service-accounts add-iam-policy-binding "${SA_EMAIL}" \
  --role="roles/iam.workloadIdentityUser" \
  --member="${POOL_RESOURCE}" \
  --project="${GCP_PROJECT_ID}"

PROVIDER_RESOURCE="projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL_ID}/providers/${PROVIDER_ID}"

echo ""
echo "══════════════════════════════════════════════════════════════"
echo " Add these as GitHub Secrets in your repository settings:"
echo "══════════════════════════════════════════════════════════════"
echo " WORKLOAD_IDENTITY_PROVIDER = ${PROVIDER_RESOURCE}"
echo " SERVICE_ACCOUNT            = ${SA_EMAIL}"
echo " GCP_PROJECT_ID             = ${GCP_PROJECT_ID}"
echo "══════════════════════════════════════════════════════════════"
