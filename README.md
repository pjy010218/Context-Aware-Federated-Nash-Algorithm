# Context-Aware-Federated-Nash-Algorithm
The CAFN Algorithm is a new paradigm in Federated Learning, taking in consideration of the global model loss, thus creating a Nash Equilibrium of all domains.

---

### Key Components
1. Theta (θ): Global backbone (e.g., shared encoder or foundation model)
2. Phi-i (ϕᵢ): Client-specific head or prompt (e.g., task head, adapter, prompt vector)
3. Alpha-i (αᵢ): Cooperation parameter — blends local vs. global loss
4. Loss_blend (L_blend): Weighted loss combining local + estimated global impact
5. Theta-i (Δθᵢ): The client's update to the global model
6. C-i (cᵢ): Context vector encoding client task or distribution
7. a, c-i (a(cᵢ)): Attention-like weight for update aggregation, based on context