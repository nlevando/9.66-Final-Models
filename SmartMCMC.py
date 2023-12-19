import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS


data_smart = [
    30, 30, 27, 26, 28, 30, 26, 30, 30, 28, 33, 25, 30, 28, 32,
    28, 28, 30, 26, 26, 29, 25, 28, 30, 29, 27, 28, 27, 30, 27,
    28, 27, 27, 30, 27, 28, 30, 30, 30, 25
]

# Convert the list to a tensor for Pyro
scores = torch.tensor(data_smart, dtype=torch.float32)

# Define the model for "smart"
def model_smart(scores):
    mu = pyro.sample("mu", dist.Normal(20.0, 10.0))  # Prior for the mean
    sigma = pyro.sample("sigma", dist.Uniform(0.0, 10.0))  # Prior for the standard deviation
    with pyro.plate("data", len(scores)):
        pyro.sample("obs", dist.Normal(mu, sigma), obs=scores)

# Run MCMC for "smart"
nuts_kernel = NUTS(model_smart)
mcmc = MCMC(nuts_kernel, num_samples=1000, warmup_steps=200)
mcmc.run(scores)

# Extracting the results
posterior_samples = mcmc.get_samples()

# Visualize the posterior distribution of the parameters
import matplotlib.pyplot as plt
import seaborn as sns

mu_samples = posterior_samples["mu"].numpy()
sigma_samples = posterior_samples["sigma"].numpy()

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.histplot(mu_samples, kde=True, color='skyblue', bins=30)
plt.title('Posterior Distribution of Mean (Mu)')

plt.subplot(1, 2, 2)
sns.histplot(sigma_samples, kde=True, color='salmon', bins=30)
plt.title('Posterior Distribution of Standard Deviation (Sigma)')

plt.tight_layout()
plt.show()

# KDE plot for the estimated probability density function of ACT scores
mu_mean = torch.mean(posterior_samples["mu"])
sigma_mean = torch.mean(posterior_samples["sigma"])
scores_range = torch.linspace(min(scores).item(), max(scores).item(), 100)
pdf_values = dist.Normal(mu_mean, sigma_mean).log_prob(scores_range).exp().numpy()

plt.figure(figsize=(10, 5))
sns.lineplot(x=scores_range.numpy(), y=pdf_values, color='blue')
plt.title('Estimated Probability Density Function for "Smart"')
plt.xlabel('ACT Score')
plt.ylabel('Density')
plt.show()
