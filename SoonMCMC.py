



import torch
import pyro
import pyro.distributions as dist
import pandas as pd
from pyro.infer import MCMC, NUTS
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np




# Data preparation
data = [
    {"context": "transport", "expectedTime": 23.49},
    {"context": "transport", "expectedTime": 5.39},
    {"context": "delivery", "expectedTime": 5.47},
    {"context": "natural_events", "expectedTime": 5.43},
    {"context": "delivery", "expectedTime": 4.45},
    {"context": "delivery", "expectedTime": 6.0},
    {"context": "natural_events", "expectedTime": 9.78},
    {"context": "delivery", "expectedTime": 2.9},
    {"context": "delivery", "expectedTime": 4.94},
    {"context": "transport", "expectedTime": 9.66},
    {"context": "natural_events", "expectedTime": 5.54},
    {"context": "transport", "expectedTime": 23.24},
    {"context": "natural_events", "expectedTime": 1.49},
    {"context": "delivery", "expectedTime": 6.0},
    {"context": "delivery", "expectedTime": 4.9},
    {"context": "delivery", "expectedTime": 5.18},
    {"context": "natural_events", "expectedTime": 9.19},
    {"context": "delivery", "expectedTime": 5.66},
    {"context": "transport", "expectedTime": 10.91},
    {"context": "transport", "expectedTime": 6.77},
    {"context": "transport", "expectedTime": 8.41},
    {"context": "natural_events", "expectedTime": 1.4},
    {"context": "natural_events", "expectedTime": 2.94},
    {"context": "natural_events", "expectedTime": 3.71},
    {"context": "transport", "expectedTime": 15.96},
    {"context": "transport", "expectedTime": 19.97},
    {"context": "transport", "expectedTime": 15.76},
    {"context": "delivery", "expectedTime": 4.58},
    {"context": "delivery", "expectedTime": 2.94},
    {"context": "natural_events", "expectedTime": 7.37}
]
df = pd.DataFrame(data)
context_encoding = {context: i for i, context in enumerate(df['context'].unique())}
df['context_code'] = df['context'].apply(lambda x: context_encoding[x])

# Analyze data to inform priors
context_stats = df.groupby('context')['expectedTime'].agg(['mean', 'std']).to_dict()

# Bayesian model with context-specific priors
def model(context_codes, expected_times):
    unique_contexts = len(context_encoding)

    means = torch.zeros(unique_contexts)
    variances = torch.zeros(unique_contexts)

    for i in range(unique_contexts):
        context_name = list(context_encoding.keys())[i]

        # Mean prior
        mean_prior = dist.Normal(context_stats['mean'][context_name], context_stats['std'][context_name])
        means[i] = pyro.sample(f"mean_{context_name}", mean_prior)

        # Variance prior - using Half-Normal
        variance_prior = dist.HalfNormal(context_stats['std'][context_name])
        variances[i] = pyro.sample(f"variance_{context_name}", variance_prior)

    with pyro.plate("data", len(expected_times)):
        context_means = means[context_codes]
        context_variances = variances[context_codes]
        pyro.sample("obs", dist.Normal(context_means, context_variances.sqrt()), obs=expected_times)

# Running MCMC
context_codes = torch.tensor(df['context_code'].values, dtype=torch.long)
expected_times = torch.tensor(df['expectedTime'].values, dtype=torch.float32)

nuts_kernel = NUTS(model)
mcmc = MCMC(nuts_kernel, num_samples=1000, warmup_steps=500)
mcmc.run(context_codes, expected_times)

# Extracting and analyzing the results
posterior_samples = mcmc.get_samples()

# For each context

for i, context in enumerate(context_encoding.keys()):
    # Extract the mean and variance for each context
    mean = posterior_samples[f'mean_{context}'].mean().item()  # Ensuring mean is a scalar
    variance = posterior_samples[f'variance_{context}'].mean().item()  # Ensuring variance is a scalar
    std = np.sqrt(variance)

    # Generate a range of time values
    time_values = np.linspace(mean - 3*std, mean + 3*std, 100)
    time_values_tensor = torch.tensor(time_values, dtype=torch.float32)

    # Ensure the mean and std are also tensors
    mean_tensor = torch.tensor(mean, dtype=torch.float32)
    std_tensor = torch.tensor(std, dtype=torch.float32)

    # Get the probability density for each value
    density = dist.Normal(mean_tensor, std_tensor).log_prob(time_values_tensor).exp().numpy()

    # Plotting
    plt.figure(figsize=(8, 4))
    sns.lineplot(x=time_values, y=density, label=f'Context: {context}')
    plt.title(f'Probability Density of "Soon" for {context}')
    plt.xlabel('Time')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.show()