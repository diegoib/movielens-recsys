variable "project_id"    { type = string }
variable "region"        { type = string }
variable "repository_id" {
  type    = string
  default = "movielens-recsys"
}

resource "google_artifact_registry_repository" "docker" {
  project       = var.project_id
  location      = var.region
  repository_id = var.repository_id
  format        = "DOCKER"
  description   = "Docker images for movielens-recsys"
}

output "repository_url" {
  value = "${var.region}-docker.pkg.dev/${var.project_id}/${var.repository_id}"
}
