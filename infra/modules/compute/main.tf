variable "project_id" { type = string }
variable "region" { type = string }
variable "zone" { type = string }
variable "streaming_vm_sa" {
  type        = string
  description = "Service account email to attach to the streaming VM"
}

variable "datagen_vm_sa" {
  type        = string
  description = "Service account email to attach to the datagen VM"
}

# ── Streaming VM (e2-standard-2, preemptible) ────────────────────────────────

resource "google_compute_instance" "streaming_vm" {
  name           = "streaming-vm"
  project        = var.project_id
  zone           = var.zone
  machine_type   = "e2-standard-2"
  desired_status = "TERMINATED"

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2204-lts"
      size  = 30
    }
  }

  # Preemptible: can be reclaimed by GCP at any time; startup script recovers state
  scheduling {
    preemptible         = true
    automatic_restart   = false
    on_host_maintenance = "TERMINATE"
  }

  network_interface {
    network = "default"
    access_config {
      nat_ip = google_compute_address.streaming_vm_static_ip.address
    }
  }

  service_account {
    email  = var.streaming_vm_sa
    scopes = ["cloud-platform"]
  }

  metadata_startup_script = <<-EOT
    #!/bin/bash
    set -euo pipefail
    apt-get clean
    rm -rf /var/lib/apt/lists/*
    apt-get update -q
    apt-get install -y ca-certificates curl git openssh-server
    systemctl enable ssh
    systemctl start ssh

    # Install Docker CE from the official Docker repository
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
      -o /etc/apt/keyrings/docker.asc
    chmod a+r /etc/apt/keyrings/docker.asc
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
      https://download.docker.com/linux/ubuntu \
      $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
      | tee /etc/apt/sources.list.d/docker.list
    apt-get update -q
    apt-get install -y docker-ce docker-ce-cli containerd.io \
      docker-buildx-plugin docker-compose-plugin

    systemctl enable docker
    systemctl start docker
    usermod -aG docker ubuntu
    echo "GCP_PROJECT_ID=${var.project_id}" > /root/movielens-recsys/.env
  EOT

  tags = ["streaming-vm"]
}

# ── Airflow VM (e2-micro, free tier) ──────────────────────────────────────────

resource "google_compute_instance" "airflow_vm" {
  name           = "airflow-vm"
  project        = var.project_id
  zone           = var.zone
  machine_type   = "e2-micro"
  desired_status = "TERMINATED"

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

# ── Datagen VM (e2-highmem-4, stopped by default) ────────────────────────────

resource "google_compute_instance" "datagen_vm" {
  name           = "datagen-vm"
  project        = var.project_id
  zone           = var.zone
  machine_type   = "e2-standard-8"
  desired_status = "TERMINATED"

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2204-lts"
      size  = 50
    }
  }

  network_interface {
    network = "default"
    access_config {}
  }

  service_account {
    email  = var.datagen_vm_sa
    scopes = ["cloud-platform"]
  }

  metadata_startup_script = <<-EOT
    #!/bin/bash
    set -euo pipefail
    apt-get update -q
    apt-get install -y python3 python3-pip git curl tmux
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ln -sf /root/.local/bin/uv /usr/local/bin/uv
  EOT

  tags = ["datagen-vm"]
}

# ── Static IP for streaming VM ────────────────────────────────────────────────

resource "google_compute_address" "streaming_vm_static_ip" {
  name    = "streaming-vm-static-ip"
  project = var.project_id
  region  = var.region
}

# ── Firewall rules ─────────────────────────────────────────────────────────────

# Allow RedPanda Console (UI) from anywhere
resource "google_compute_firewall" "allow_redpanda_console" {
  name    = "allow-redpanda-console-8080"
  project = var.project_id
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["8080"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["streaming-vm"]
}

# Allow MLflow UI + tracking API from anywhere (Cloud Run Jobs need this; port forward for local UI)
resource "google_compute_firewall" "allow_mlflow" {
  name    = "allow-mlflow-5000"
  project = var.project_id
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["5000"]
  }

  source_ranges = ["0.0.0.0/0"] # educational project; restrict to VPC in production
  target_tags   = ["streaming-vm"]
}


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

output "streaming_vm_static_ip" {
  value = google_compute_address.streaming_vm_static_ip.address
}

output "streaming_vm_external_ip" {
  value = google_compute_instance.streaming_vm.network_interface[0].access_config[0].nat_ip
}

output "streaming_vm_internal_ip" {
  value = google_compute_instance.streaming_vm.network_interface[0].network_ip
}

output "datagen_vm_external_ip" {
  value = google_compute_instance.datagen_vm.network_interface[0].access_config[0].nat_ip
}
