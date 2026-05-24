variable "project_id" { type = string }
variable "region"     { type = string }

resource "google_storage_bucket" "data" {
  name          = "${var.project_id}-data"
  project       = var.project_id
  location      = var.region
  force_destroy = false

  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  # Delete noncurrent (archived) object versions older than 30 days
  lifecycle_rule {
    condition {
      age        = 30
      with_state = "ARCHIVED"
    }
    action {
      type = "Delete"
    }
  }
}

output "bucket_name" {
  value = google_storage_bucket.data.name
}

output "bucket_url" {
  value = "gs://${google_storage_bucket.data.name}"
}
