variable "project_id"  { type = string }
variable "region"      { type = string }
variable "serving_sa"  {
  type        = string
  description = "Service account email for the serving Cloud Run service"
}
variable "redis_host" {
  type        = string
  description = "Internal IP of streaming VM (Redis host)"
}
variable "model_dir" {
  type        = string
  description = "GCS path to model artifacts (fallback when MLflow has no Production model)"
}
variable "mlflow_uri" {
  type        = string
  description = "MLflow tracking server URL, e.g. http://<streaming-vm-ip>:5000"
}
variable "redpanda_brokers" {
  type        = string
  description = "Kafka broker address for RedPanda, e.g. <streaming-vm-internal-ip>:9092"
}

resource "google_cloud_run_v2_service" "recsys_serving" {
  name     = "recsys-serving"
  project  = var.project_id
  location = var.region

  deletion_protection = false

  template {
    service_account                  = var.serving_sa
    max_instance_request_concurrency = 80

    scaling {
      min_instance_count = 0
      max_instance_count = 3
    }

    # VPC Direct Egress so Cloud Run can reach Redis on the streaming VM internal IP
    vpc_access {
      network_interfaces {
        network = "default"
      }
      egress = "PRIVATE_RANGES_ONLY"
    }

    containers {
      image = "us-docker.pkg.dev/cloudrun/container/hello" # replaced via make serve-deploy
      resources {
        limits = {
          cpu    = "1"
          memory = "1Gi"
        }
      }
      env {
        name  = "REDIS_HOST"
        value = var.redis_host
      }
      env {
        name  = "MODEL_DIR"
        value = var.model_dir
      }
      env {
        name  = "MLFLOW_TRACKING_URI"
        value = var.mlflow_uri
      }
      env {
        name  = "MLFLOW_MODEL_NAME"
        value = "two-tower-recsys"
      }
      env {
        name  = "REDPANDA_BROKERS"
        value = var.redpanda_brokers
      }
    }
  }

  # Ignore image changes — the image is managed by make serve-deploy, not terraform
  lifecycle {
    ignore_changes = [template[0].containers[0].image]
  }
}

# Public access — required for Simulator 2 and external clients to call the API
resource "google_cloud_run_v2_service_iam_member" "public_invoker" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.recsys_serving.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

output "service_url" {
  value = google_cloud_run_v2_service.recsys_serving.uri
}
