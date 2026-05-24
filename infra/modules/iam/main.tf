variable "project_id" { type = string }

# NOTE: the github-actions SA is managed by infra/bootstrap/setup_oidc.sh (Workload Identity
# Federation setup). It is intentionally excluded here to avoid conflicts with the OIDC script.

# ── Cloud Run serving SA ──────────────────────────────────────────────────────

resource "google_service_account" "cloud_run_serving" {
  project      = var.project_id
  account_id   = "cloud-run-serving"
  display_name = "Cloud Run Serving SA"
}

resource "google_project_iam_member" "serving_storage_viewer" {
  project = var.project_id
  role    = "roles/storage.objectViewer"
  member  = "serviceAccount:${google_service_account.cloud_run_serving.email}"
}

resource "google_project_iam_member" "serving_secret_accessor" {
  project = var.project_id
  role    = "roles/secretmanager.secretAccessor"
  member  = "serviceAccount:${google_service_account.cloud_run_serving.email}"
}

resource "google_project_iam_member" "serving_registry_reader" {
  project = var.project_id
  role    = "roles/artifactregistry.reader"
  member  = "serviceAccount:${google_service_account.cloud_run_serving.email}"
}

# ── Cloud Run jobs SA (training) ──────────────────────────────────────────────

resource "google_service_account" "cloud_run_jobs" {
  project      = var.project_id
  account_id   = "cloud-run-jobs"
  display_name = "Cloud Run Jobs SA"
}

resource "google_project_iam_member" "jobs_storage_admin" {
  project = var.project_id
  role    = "roles/storage.objectAdmin"
  member  = "serviceAccount:${google_service_account.cloud_run_jobs.email}"
}

resource "google_project_iam_member" "jobs_registry_reader" {
  project = var.project_id
  role    = "roles/artifactregistry.reader"
  member  = "serviceAccount:${google_service_account.cloud_run_jobs.email}"
}

# ── Outputs ───────────────────────────────────────────────────────────────────

output "serving_sa_email" {
  value = google_service_account.cloud_run_serving.email
}

output "jobs_sa_email" {
  value = google_service_account.cloud_run_jobs.email
}
