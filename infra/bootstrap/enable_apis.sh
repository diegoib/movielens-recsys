#!/usr/bin/env bash
# Enables all GCP APIs required by the project.
# Run once after creating the GCP project.
set -euo pipefail

: "${GCP_PROJECT_ID:?Set GCP_PROJECT_ID before running this script}"

APIS=(
  run.googleapis.com
  compute.googleapis.com
  artifactregistry.googleapis.com
  secretmanager.googleapis.com
  cloudbuild.googleapis.com
  iam.googleapis.com
  storage.googleapis.com
  iamcredentials.googleapis.com
  cloudscheduler.googleapis.com
  firestore.googleapis.com
  pubsub.googleapis.com
)

echo "Enabling ${#APIS[@]} APIs on project: ${GCP_PROJECT_ID}"
gcloud services enable "${APIS[@]}" --project "${GCP_PROJECT_ID}"
echo "All APIs enabled."
