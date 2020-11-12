import os
import torch
import pyro
import tqdm

# this is for running the notebook in our testing framework
n_steps = 1000
# pyro.enable_validation(True)

# create some data with 6 observed heads and 4 observed tails
data = []
for _ in range(6):
    data.append(torch.tensor(1.0))
for _ in range(4):
    data.append(torch.tensor(0.0))

def model(data):
    # define the hyperparameters that control the beta prior
    alpha0 = torch.tensor(10.0)
    beta0 = torch.tensor(10.0)
    # sample f from the beta prior
    f = pyro.sample("latent_fairness", pyro.distributions.Beta(alpha0, beta0))
    # loop over the observed data
    for i in range(len(data)):
        # observe datapoint i using the bernoulli likelihood
        pyro.sample("obs_{}".format(i), pyro.distributions.Bernoulli(f), obs=data[i])

def guide(data):
    # register the two variational parameters with Pyro
    # - both parameters will have initial value 15.0.
    # - because we invoke torch.distributions.constraints.positive, the optimizer
    # will take gradients on the unconstrained parameters
    # (which are related to the constrained parameters by a log)
    alpha_q = pyro.param("alpha_q", torch.tensor(15.0), constraint=torch.distributions.constraints.positive)
    beta_q = pyro.param("beta_q", torch.tensor(15.0), constraint=torch.distributions.constraints.positive)
    # sample latent_fairness from the pyro.distributionsribution Beta(alpha_q, beta_q)
    pyro.sample("latent_fairness", pyro.distributions.Beta(alpha_q, beta_q))

# setup the optimizer
optimizer = pyro.optim.Adam({"lr": 0.001, "betas": (0.90, 0.999)})

# setup the inference algorithm
pyro.infer.SVI = pyro.infer.SVI(model, guide, optimizer, loss=pyro.infer.Trace_ELBO())

# do gradient steps
for _ in tqdm.trange(n_steps):
    pyro.infer.SVI.step(data)

# grab the learned variational parameters
alpha_q = pyro.param("alpha_q").item()
beta_q = pyro.param("beta_q").item()

# here we use some facts about the beta distribution
# compute the inferred mean of the coin's fairness
inferred_mean = alpha_q / (alpha_q + beta_q)
# compute inferred standard deviation
factor = beta_q / (alpha_q * (1.0 + alpha_q + beta_q))
inferred_std = inferred_mean * factor ** .5

print("\nbased on the data and our prior belief, the fairness of the coin is %.3f +- %.3f" % (inferred_mean, inferred_std))