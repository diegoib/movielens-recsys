variable "project_id"      { type = string }
variable "region"          { type = string }
variable "zone"            { type = string }
variable "streaming_vm_sa" {
  type        = string
  description = "Service account email to attach to the streaming VM"
}

# ── Streaming VM (e2-medium, preemptible) ─────────────────────────────────────

resource "google_compute_instance" "streaming_vm" {
  name         = "streaming-vm"
  project      = var.project_id
  zone         = var.zone
  machine_type = "e2-medium"

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2204-lts"
      size  = 30
    }
  }

  # Preemptible: can be reclaimed by GCP at any time; startup script recovers state
  scheduling {
    preemptible        = true
    automatic_restart  = false
    on_host_maintenance = "TERMINATE"
  }

  network_interface {
    network = "default"
    access_config {} # ephemeral external IP
  }

  service_account {
    email  = var.streaming_vm_sa
    scopes = ["cloud-platform"]
  }

  metadata_startup_script = <<-EOT
    #!/bin/bash
    set -euo pipefail
    apt-get update -q
    apt-get install -y docker.io docker-compose-plugin git
    systemctl enable docker
    systemctl start docker
    usermod -aG docker ubuntu
    mkdir -p /opt/movielens-recsys
  EOT

  tags = ["streaming-vm"]
}

# ── Airflow VM (e2-micro, free tier) ──────────────────────────────────────────

resource "google_compute_instance" "airflow_vm" {
  name         = "airflow-vm"
  project      = var.project_id
  zone         = var.zone
  machine_type = "e2-micro"

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2204-lts"
      size  = 20
    }
  }

  network_interface {
    network = "default"
    access_config {}
  }

  metadata_startup_script = <<-EOT
    #!/bin/bash
    set -euo pipefail
    apt-get update -q
    apt-get install -y docker.io docker-compose-plugin
    systemctl enable docker
    systemctl start docker
    usermod -aG docker ubuntu
    mkdir -p /opt/airflow
  EOT

  tags = ["airflow-vm"]
}

# ── Firewall rules ─────────────────────────────────────────────────────────────

# Allow Cloud Run to reach Redis on the streaming VM via internal GCP network
resource "google_compute_firewall" "allow_redis_internal" {
  name    = "allow-redis-internal"
  project = var.project_id
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["6379"]
  }

  source_ranges = ["10.0.0.0/8", "172.16.0.0/12"]
  target_tags   = ["streaming-vm"]
}

# Allow RedPanda (Kafka API) access from internal GCP IPs
resource "google_compute_firewall" "allow_redpanda_internal" {
  name    = "allow-redpanda-internal"
  project = var.project_id
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["9092"]
  }

  source_ranges = ["10.0.0.0/8", "172.16.0.0/12"]
  target_tags   = ["streaming-vm"]
}

# ── Outputs ────────────────────────────────────────────────────────────────────

output "streaming_vm_external_ip" {
  value = google_compute_instance.streaming_vm.network_interface[0].access_config[0].nat_ip
}

output "streaming_vm_internal_ip" {
  value = google_compute_instance.streaming_vm.network_interface[0].network_ip
}
