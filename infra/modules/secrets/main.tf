variable "project_id" { type = string }

# Fill these manually after terraform apply:
#   gcloud secrets versions add kaggle-username --data-file=- <<< "your-username"
#   gcloud secrets versions add kaggle-key      --data-file=- <<< "your-api-key"

resource "google_secret_manager_secret" "kaggle_username" {
  project   = var.project_id
  secret_id = "kaggle-username"
  replication {
    auto {}
  }
}

resource "google_secret_manager_secret" "kaggle_key" {
  project   = var.project_id
  secret_id = "kaggle-key"
  replication {
    auto {}
  }
}
