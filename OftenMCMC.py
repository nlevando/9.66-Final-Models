import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Data for "often"
data_often = [
    4, 5, 3, 7, 3, 3, 5, 21, 4, 4,
    5, 5, 2, 3, 10, 4, 3, 3, 2, 5,
    3, 3, 7, 4, 5, 5, 7, 3, 4, 3,
    5, 4, 3, 3, 4, 3, 5, 4, 7, 6
]


frequencies = torch.tensor(data_often, dtype=torch.float32)


def model(frequencies):
    # Prior for the rate parameter of the Poisson distribution
    rate_prior = dist.Gamma(1, 1)  # Hyperparameters can be adjusted
    rate = pyro.sample("rate", rate_prior)

    # Sampling the observed data
    with pyro.plate("data", len(frequencies)):
        pyro.sample("obs", dist.Poisson(rate), obs=frequencies)

# Run MCMC
nuts_kernel = NUTS(model)
mcmc = MCMC(nuts_kernel, num_samples=1000, warmup_steps=200)
mcmc.run(frequencies)

# Extracting the results
posterior_samples = mcmc.get_samples()

# Extract the rate samples from the posterior
rate_samples = posterior_samples['rate'].numpy()

# Generate a range of possible rate values for plotting
rate_values = np.linspace(0, np.max(rate_samples), 200)

sns.set(style="white")  # Set the style for the plots

plt.figure(figsize=(10, 5))
sns.kdeplot(rate_samples, shade=False, color="blue", label='Posterior Density')
plt.title('Posterior Probability Density of "Often"')
plt.xlabel('Rate (number of times per week)')
plt.ylabel('Probability Density')
plt.legend()
plt.show()