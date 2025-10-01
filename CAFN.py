# ----------------------------
# Context-Aware Federated Nash (CAFN) — Pseudocode
# ----------------------------

Initialize global model θ₀
Initialize each client’s local parameters ϕᵢ₀
Initialize each client’s αᵢ ∈ [0, 1]  # selfishness vs. cooperation
Initialize context encoder or define method to generate context vector cᵢ
Set learning rate η and number of rounds T

for each communication round t = 0 to T - 1:
    
    Server broadcasts current global model θₜ
    
    Select a subset of clients (e.g., random sampling or importance-based)
    
    CLIENT-SIDE (for each selected client i):

        Receive global model θₜ

        θᵢ ← θₜ.copy()  # make a local copy for updates

        for local step k = 1 to K:
            
            Sample (x, y) ∼ Dᵢ  # local data of client i

            # Forward pass through model using θᵢ and ϕᵢ
            ŷ = f(x; θᵢ, ϕᵢ)
            
            Compute local loss:     Lᵢ = ℓ(ŷ, y)

            Estimate global loss impact: 
                Ĥₗₒ_b = Estimate_Lglob_Effect(θᵢ, θₜ)  
                # e.g., linearized change in global loss, historical gradient impact

            # Blended loss: αᵢ * local + (1 - αᵢ) * global
            L_blend = αᵢ * Lᵢ + (1 - αᵢ) * Ĥₗₒ_b

            Update θᵢ and ϕᵢ via gradient descent on L_blend

        Compute delta update:      Δθᵢ = θᵢ - θₜ

        Generate context vector:   cᵢ = Generate_Context_Vector(Dᵢ, ϕᵢ)
            # could be a learned embedding, label distribution, domain ID, etc.

        Send (Δθᵢ, cᵢ) to the server (ϕᵢ stays local)

    SERVER-SIDE:

        Receive all updates (Δθᵢ, cᵢ) from selected clients

        For each client i:
            Compute attention weight a(cᵢ)
                # This can be:
                # - Learned via dot-product or MLP on context vector
                # - Similarity-based (e.g., cosine with global context)
                # - Uniform weights (FedAvg) if context is not used

        Normalize attention weights so that ∑ₖ a(cₖ) = 1

        Aggregate global update:
            θₜ₊₁ = θₜ + η * ∑ₖ a(cₖ) * Δθₖ

        Optionally:
            - Update clustering of clients
            - Adjust αᵢ values based on incentives or performance
            - Update context encoder parameters if trainable

# End of communication rounds
