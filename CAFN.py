"""
Context-Aware Federated Nash (CAFN) Algorithm
Author: Junyeong Park
Last Updated: October 1st, 2025
Description: A flexible implementation template of CAFN for federated learning with context-aware client aggregation.
"""

import random
import numpy as np

# -----------------------------------------------------------------------------
# Placeholder model and utility functions (To be replaced with actual ML code)
# -----------------------------------------------------------------------------

def initialize_global_model():
    """Initialize global model θ (e.g., random weights or pre-trained backbone)."""
    return np.random.randn(100)

def initialize_local_head():
    """Initialize local model head or prompt ϕᵢ."""
    return np.random.randn(10)

def compute_local_loss(x, y, theta, phi):
    """Compute local loss ℓ(f(x; θ, ϕ), y) - Placeholder function."""
    return np.linalg.norm(np.dot(theta[:10], x) + np.dot(phi, x) - y)

def estimate_global_loss(theta_i, theta_t):
    """Estimate the global impact of a local update - Placeholder (linear approximation)."""
    return np.linalg.norm(theta_i - theta_t)  # dummy surrogate

def generate_context_vector(data, phi):
    """Generate a context vector cᵢ from client data or local head."""
    return np.mean(data, axis=0) + 0.1 * phi  # simple placeholder

def compute_attention_weight(context_vector):
    """Compute attention weight a(cᵢ) from context vector (e.g., norm-based)."""
    return np.linalg.norm(context_vector)

# -----------------------------------------------------------------------------
# Client class representing a single participant in FL
# -----------------------------------------------------------------------------

class Client:
    def __init__(self, client_id, alpha, data):
        self.id = client_id
        self.alpha = alpha  # selfishness vs cooperation
        self.phi = initialize_local_head()
        self.data = data  # (x, y) pairs
        self.theta = None  # will be set each round
    
    def local_update(self, global_theta, K=5, lr=0.01):
        self.theta = global_theta.copy()

        for _ in range(K):
            x, y = random.choice(self.data)
            local_loss = compute_local_loss(x, y, self.theta, self.phi)
            global_estimate = estimate_global_loss(self.theta, global_theta)

            blended_loss = self.alpha * local_loss + (1 - self.alpha) * global_estimate

            # Dummy gradient step (placeholder logic)
            grad_theta = 0.1 * (self.theta - global_theta)
            grad_phi = 0.1 * self.phi

            self.theta -= lr * grad_theta
            self.phi -= lr * grad_phi

        delta_theta = self.theta - global_theta
        context = generate_context_vector([x for (x, _) in self.data], self.phi)

        return delta_theta, context

# -----------------------------------------------------------------------------
# CAFN Federated Learning Procedure
# -----------------------------------------------------------------------------

def CAFN(num_rounds=10, num_clients=5, client_frac=0.6, eta=0.1):
    # Initialize global model
    theta = initialize_global_model()

    # Simulate client data
    clients = []
    for i in range(num_clients):
        # Simulate small dataset per client
        data = [(np.random.randn(10), np.random.randn()) for _ in range(20)]
        alpha = random.uniform(0.3, 1.0)  # local vs. global emphasis
        clients.append(Client(client_id=i, alpha=alpha, data=data))

    for t in range(num_rounds):
        print(f"--- Round {t+1} ---")

        selected_clients = random.sample(clients, int(client_frac * num_clients))
        updates = []
        weights = []

        for client in selected_clients:
            delta_theta, context_vector = client.local_update(global_theta=theta)

            weight = compute_attention_weight(context_vector)
            updates.append((delta_theta, weight))

        # Normalize attention weights
        total_weight = sum(w for _, w in updates)
        if total_weight == 0:
            total_weight = 1e-6  # avoid divide by zero
        updates = [(delta, w / total_weight) for (delta, w) in updates]

        # Aggregate global model
        aggregated_update = sum(w * delta for delta, w in updates)
        theta += eta * aggregated_update

        print(f"Updated global model norm: {np.linalg.norm(theta):.4f}")

# -----------------------------------------------------------------------------
# Run CAFN
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    CAFN(num_rounds=20, num_clients=10)
