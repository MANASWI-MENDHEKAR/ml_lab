# monte_carlo_gbm.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def simulate_gbm(S0, mu, sigma, T=1.0, steps=252, n_sims=1000):
    dt = T/steps
    sims = np.zeros((n_sims, steps+1))
    sims[:,0] = S0
    for t in range(1, steps+1):
        z = np.random.randn(n_sims)
        sims[:,t] = sims[:,t-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z)
    return sims

# Example:
# S0 = 100; mu = 0.08; sigma = 0.2
# sims = simulate_gbm(S0, mu, sigma, T=1, steps=252, n_sims=5000)
# predicted_price = sims[:,-1].mean()
# Create a synthetic dataset
def create_synthetic_data(n_samples=1000):
    np.random.seed(42)  # For reproducibility
    S0 = 100
    mu = 0.08
    sigma = 0.2
    T = 1.0
    steps = 252
    n_sims = n_samples

    sims = simulate_gbm(S0, mu, sigma, T, steps, n_sims)
    return pd.DataFrame(sims)

# Generate the synthetic dataset
synthetic_data = create_synthetic_data(5000)
print(synthetic_data.head())

plt.figure(figsize=(10, 5))
for i in range(10):      # Plot only first 10 paths for clarity
    plt.plot(synthetic_data.iloc[i], alpha=0.7)

plt.title("Monte Carlo Simulation of GBM Stock Price Paths")
plt.xlabel("Time Steps")
plt.ylabel("Stock Price")
plt.grid(True)
plt.show()