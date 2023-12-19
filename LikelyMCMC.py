import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torch.distributions import Beta


data_likely = [
    0.8, 0.8, 0.75, 0.9, 0.75, 0.8, 0.75, 0.75, 0.8, 0.85,
    0.9, 0.75, 0.85, 0.8, 0.65, 0.85, 0.8, 0.85, 0.5, 0.6,
    0.7, 0.7, 0.5, 0.75, 0.75, 0.7, 0.8, 0.8, 0.6, 0.8,
    0.75, 0.8, 0.8, 0.8, 0.7, 0.8, 0.9, 0.75, 0.75, 0.5
]

# Convert the list to a tensor for Pyro
probabilities = torch.tensor(data_likely, dtype=torch.float32)

# Define the model for "likely"
def model_likely(probabilities):
    # Prior for the alpha and beta parameters of the Beta distribution
    alpha_prior = pyro.sample("alpha", dist.Uniform(0.5, 5))
    beta_prior = pyro.sample("beta", dist.Uniform(0.5, 5))
    
    with pyro.plate("data", len(probabilities)):
        pyro.sample("obs", dist.Beta(alpha_prior, beta_prior), obs=probabilities)

# Run MCMC
nuts_kernel = NUTS(model_likely)
mcmc = MCMC(nuts_kernel, num_samples=1000, warmup_steps=500)
mcmc.run(probabilities)

# Extracting the results
posterior_samples = mcmc.get_samples()


alpha_samples = posterior_samples["alpha"].numpy()
beta_samples = posterior_samples["beta"].numpy()


plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.kdeplot(alpha_samples, shade=True, color='skyblue', label='Posterior alpha')
plt.title('Posterior Probability Density of Alpha')
plt.xlabel('Alpha')
plt.ylabel('Density')
plt.legend()

# Plotting the KDE for the beta parameter
plt.subplot(1, 2, 2)
sns.kdeplot(beta_samples, shade=True, color='salmon', label='Posterior beta')
plt.title('Posterior Probability Density of Beta')
plt.xlabel('Beta')
plt.ylabel('Density')
plt.legend()

plt.tight_layout()
plt.show()


alpha_samples = posterior_samples['alpha'].numpy()
beta_samples = posterior_samples['beta'].numpy()


alpha_mean = np.mean(alpha_samples)
beta_mean = np.mean(beta_samples)

# Create a range of probability values
probability_range = np.linspace(0, 1, 100)

# Calculate the probability density function (PDF) for the Beta distribution
pdf_values = Beta(torch.tensor([alpha_mean]), torch.tensor([beta_mean])).log_prob(torch.tensor(probability_range)).exp().numpy()

# Plotting the PDF
plt.figure(figsize=(10, 5))
sns.lineplot(x=probability_range, y=pdf_values, color='blue')
plt.title('Posterior Probability Density Function for "Likely"')
plt.xlabel('Probability')
plt.ylabel('Density')
plt.show()
