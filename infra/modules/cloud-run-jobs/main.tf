variable "project_id" { type = string }
variable "region"     { type = string }
variable "jobs_sa" {
  type        = string
  description = "Service account email for the training job"
}

variable "mlflow_uri" {
  type        = string
  description = "MLflow tracking server URI reachable from Cloud Run (http://<static-ip>:5000)"
}

variable "gcs_data_path" {
  type        = string
  description = "gs:// URI to train_dataset.parquet"
}

variable "gcs_movies_path" {
  type        = string
  description = "gs:// URI to movies.csv (needed to build genre vocab)"
}

variable "gcs_onnx_out" {
  type        = string
  description = "gs:// URI where the exported ONNX model will be written"
}

variable "image" {
  type        = string
  default     = "us-docker.pkg.dev/cloudrun/container/hello"
  description = "Docker image for the training job; updated by make docker-build-training"
}

resource "google_cloud_run_v2_job" "training" {
  name     = "training-job"
  project  = var.project_id
  location = var.region

  template {
    template {
      service_account = var.jobs_sa
      timeout         = "21600s" # 6 hours max

      containers {
        image = var.image
        resources {
          limits = {
            cpu    = "4"
            memory = "16Gi"
          }
        }
        env {
          name  = "MLFLOW_TRACKING_URI"
          value = var.mlflow_uri
        }
        env {
          name  = "GCS_DATA_PATH"
          value = var.gcs_data_path
        }
        env {
          name  = "GCS_MOVIES_PATH"
          value = var.gcs_movies_path
        }
        env {
          name  = "GCS_ONNX_OUTPUT"
          value = var.gcs_onnx_out
        }
      }
    }
  }

  # Image is managed by make docker-build-training + make train-gcp, not by terraform
  lifecycle {
    ignore_changes = [template[0].template[0].containers[0].image]
  }
}

output "job_name" {
  value = google_cloud_run_v2_job.training.name
}
