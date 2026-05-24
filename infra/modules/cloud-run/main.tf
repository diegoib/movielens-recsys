variable "project_id"  { type = string }
variable "region"      { type = string }
variable "serving_sa"  {
  type        = string
  description = "Service account email for the serving Cloud Run service"
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

    containers {
      image = "us-docker.pkg.dev/cloudrun/container/hello" # replaced in Phase 6 via make serve-deploy
      resources {
        limits = {
          cpu    = "1"
          memory = "512Mi"
        }
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
