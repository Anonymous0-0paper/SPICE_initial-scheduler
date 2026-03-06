#!/bin/bash
set -eou pipefail

# --- DEFAULTS ---
DEFAULT_THANOS_PORT="31218"

# Initialize variables
CLUSTER_NAME=""
HOST_IP=""
THANOS_PORT="$DEFAULT_THANOS_PORT"

# --- HELPER FUNCTIONS ---
log_info() { echo -e "\033[34m[INFO]\033[0m $1"; }
log_success() { echo -e "\033[32m[SUCCESS]\033[0m $1"; }
log_error() { echo -e "\033[31m[ERROR]\033[0m $1"; }
log_warn() { echo -e "\033[33m[WARN]\033[0m $1"; }

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -c <name>    Member Cluster Name (Mandatory)"
    echo "  -i <ip>      Karmada Host IP (Mandatory)"
    echo "  -p <port>    Thanos Write Port (Default: $DEFAULT_THANOS_PORT)"
    echo "  -h           Show this help message"
    exit 1
}

# --- 1. PARSE CLI ARGUMENTS ---
while getopts ":c:i:p:m:n:h" opt; do
  case ${opt} in
    c) CLUSTER_NAME="$OPTARG" ;;
    i) HOST_IP="$OPTARG" ;;
    p) THANOS_PORT="$OPTARG" ;;
    h) usage ;;
    \?) log_error "Invalid option: -$OPTARG"; usage ;;
    :) log_error "Option -$OPTARG requires an argument."; usage ;;
  esac
done

# --- 2. INTERACTIVE FALLBACK (Prompt if missing) ---

# Cluster Name (Mandatory)
if [[ -z "$CLUSTER_NAME" ]]; then
    read -p "Enter Member Cluster Name (e.g., member1): " CLUSTER_NAME
fi
if [[ -z "$CLUSTER_NAME" ]]; then log_error "Cluster Name is required."; exit 1; fi

# Karmada Host IP (Mandatory)
if [[ -z "$HOST_IP" ]]; then
    read -p "Enter Karmada Host IP (for Thanos Remote Write): " HOST_IP
fi
if [[ -z "$HOST_IP" ]]; then log_error "Host IP is required."; exit 1; fi

# Optional Prompts
if [[ $OPTIND -eq 1 ]]; then
    # No flags passed, so let's allow interactive override of defaults
    read -p "Enter Thanos Write Port [Default: $DEFAULT_THANOS_PORT]: " INPUT_PORT
    THANOS_PORT=${INPUT_PORT:-$DEFAULT_THANOS_PORT}

fi

log_info "Configuration :: Cluster: $CLUSTER_NAME |"
log_info "Thanos Remote Write :: http://$HOST_IP:$THANOS_PORT/api/v1/receive"


# --- 3. INSTALL PROMETHEUS ---
log_info "Setting up Prometheus..."

# Ensure namespace exists (idempotent)
kubectl create namespace monitoring --dry-run=client -o yaml | kubectl apply -f -

helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update > /dev/null

# Install/Upgrade Prometheus
helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
  -n monitoring \
  --set prometheus.prometheusSpec.externalLabels.cluster="$CLUSTER_NAME" \
  --set prometheus.prometheusSpec.remoteWrite[0].url="http://$HOST_IP:$THANOS_PORT/api/v1/receive" \
  --set prometheus.prometheusSpec.scrapeInterval="1s" \
  --set prometheus.prometheusSpec.evaluationInterval="1s" \
  --wait

# --- 4. EXPOSE PROMETHEUS (NodePort) ---
log_info "Exposing Prometheus via NodePort..."

cat <<EOF | kubectl apply -n monitoring -f -
apiVersion: v1
kind: Service
metadata:
  name: prometheus-nodeport
spec:
  type: NodePort
  ports:
  - name: web
    nodePort: 30000
    port: 9090
    protocol: TCP
    targetPort: 9090
  selector:
    app.kubernetes.io/name: prometheus
    prometheus: prometheus-kube-prometheus-prometheus
EOF

log_success "Installation Complete! Monitoring stack is ready on $CLUSTER_NAME."

