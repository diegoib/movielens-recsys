terraform {
  required_version = ">= 1.9"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }
  # State stored locally in terraform.tfstate (gitignored)
}

variable "project_id" {
  type        = string
  description = "GCP project ID"
}

variable "region" {
  type        = string
  default     = "us-central1"
  description = "GCP region for all resources"
}

variable "zone" {
  type        = string
  default     = "us-central1-a"
  description = "GCP zone for compute instances (must be within region)"
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# ── Required APIs ──────────────────────────────────────────────────────────────
# Enabling here ensures all modules can assume the APIs are available.

resource "google_project_service" "artifact_registry" {
  project            = var.project_id
  service            = "artifactregistry.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "run" {
  project            = var.project_id
  service            = "run.googleapis.com"
  disable_on_destroy = false
}

module "gcs" {
  source     = "./modules/gcs"
  project_id = var.project_id
  region     = var.region
}

module "artifact_registry" {
  source     = "./modules/artifact-registry"
  project_id = var.project_id
  region     = var.region

  depends_on = [google_project_service.artifact_registry]
}

module "iam" {
  source     = "./modules/iam"
  project_id = var.project_id
}

module "compute" {
  source          = "./modules/compute"
  project_id      = var.project_id
  region          = var.region
  zone            = var.zone
  streaming_vm_sa = module.iam.jobs_sa_email
  datagen_vm_sa   = module.iam.jobs_sa_email
}

module "cloud_run" {
  source     = "./modules/cloud-run"
  project_id = var.project_id
  region     = var.region
  serving_sa = module.iam.serving_sa_email
  redis_host = module.compute.streaming_vm_internal_ip
  model_dir  = "gs://${module.gcs.bucket_name}/models"
  mlflow_uri       = "http://${module.compute.streaming_vm_static_ip}:5000"
  redpanda_brokers = "${module.compute.streaming_vm_internal_ip}:9092"
}

module "cloud_run_jobs" {
  source        = "./modules/cloud-run-jobs"
  project_id    = var.project_id
  region        = var.region
  jobs_sa       = module.iam.jobs_sa_email
  mlflow_uri    = "http://${module.compute.streaming_vm_static_ip}:5000"
  gcs_data_path   = "gs://${module.gcs.bucket_name}/processed/train_dataset.parquet"
  gcs_movies_path = "gs://${module.gcs.bucket_name}/raw/movies.csv"
  gcs_onnx_out    = "gs://${module.gcs.bucket_name}/models/model.onnx"

  depends_on = [google_project_service.artifact_registry, google_project_service.run]
}

module "secrets" {
  source     = "./modules/secrets"
  project_id = var.project_id
}

output "datagen_vm_ip"         { value = module.compute.datagen_vm_external_ip }
output "bucket_name"           { value = module.gcs.bucket_name }
output "bucket_url"            { value = module.gcs.bucket_url }
output "repository_url"        { value = module.artifact_registry.repository_url }
output "streaming_vm_ip"       { value = module.compute.streaming_vm_external_ip }
output "streaming_vm_static_ip" { value = module.compute.streaming_vm_static_ip }
output "streaming_vm_internal" { value = module.compute.streaming_vm_internal_ip }
output "serving_url"           { value = module.cloud_run.service_url }
