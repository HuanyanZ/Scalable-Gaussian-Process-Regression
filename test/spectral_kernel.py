import math
import torch
import gpytorch
from matplotlib import pyplot as plt
import os
from torch.distributions import Distribution
from torch.utils import benchmark


class SpectralDeltaGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, num_deltas, noise_init=None):
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(1e-11)
        )
        likelihood.register_prior(
            "noise_prior", gpytorch.priors.HorseshoePrior(0.1), "noise"
        )
        likelihood.noise = 1e-2

        super(SpectralDeltaGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        base_covar_module = gpytorch.kernels.SpectralDeltaKernel(
            num_dims=train_x.size(-1),
            num_deltas=num_deltas,
        )
        base_covar_module.initialize_from_data(train_x[0], train_y[0])
        self.covar_module = gpytorch.kernels.ScaleKernel(base_covar_module)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


train_x, train_y, test_x, test_y = torch.load(
    "./data/BART_sample.pt", map_location="cpu"
)

if torch.cuda.is_available():
    train_x, train_y, test_x, test_y = (
        train_x.cuda(),
        train_y.cuda(),
        test_x.cuda(),
        test_y.cuda(),
    )

print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

train_x_min = train_x.min()
train_x_max = train_x.max()

train_x = train_x - train_x_min
test_x = test_x - train_x_min

train_y_mean = train_y.mean(dim=-1, keepdim=True)
train_y_std = train_y.std(dim=-1, keepdim=True)

train_y = (train_y - train_y_mean) / train_y_std
test_y = (test_y - train_y_mean) / train_y_std


model = SpectralDeltaGP(train_x, train_y, num_deltas=1500)

if torch.cuda.is_available():
    model = model.cuda()

model.train()
mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[40])

num_iters = 1000

with gpytorch.settings.max_cholesky_size(0):  # Ensure we dont try to use Cholesky
    for i in range(num_iters):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        if train_x.dim() == 3:
            loss = loss.mean()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(
                f"Iteration {i} - loss = {loss:.2f} - noise = {model.likelihood.noise.item():e}"
            )

        scheduler.step()


# Get into evaluation (predictive posterior) mode
model.eval()

# Test points are regularly spaced along [0,1]
# Make predictions by feeding model through likelihood
with torch.no_grad(), gpytorch.settings.max_cholesky_size(
    0
), gpytorch.settings.fast_pred_var():
    test_x_f = torch.cat([train_x, test_x], dim=-2)
    observed_pred = model.likelihood(model(test_x_f))
    varz = observed_pred.variance

_task = 3

plt.subplots(figsize=(15, 15), sharex=True, sharey=True)
for _task in range(2):
    ax = plt.subplot(3, 1, _task + 1)

    with torch.no_grad():
        # Initialize plot
        #         f, ax = plt.subplots(1, 1, figsize=(16, 12))

        # Get upper and lower confidence bounds
        lower = observed_pred.mean - varz.sqrt() * 1.98
        upper = observed_pred.mean + varz.sqrt() * 1.98
        lower = lower[_task]  #  + weight * test_x_f.squeeze()
        upper = upper[_task]  # + weight * test_x_f.squeeze()

        # Plot training data as black stars
        ax.plot(
            train_x[_task].detach().cpu().numpy(),
            train_y[_task].detach().cpu().numpy(),
            "k*",
        )
        ax.plot(
            test_x[_task].detach().cpu().numpy(),
            test_y[_task].detach().cpu().numpy(),
            "r*",
        )
        # Plot predictive means as blue line
        ax.plot(
            test_x_f[_task].detach().cpu().numpy(),
            (observed_pred.mean[_task]).detach().cpu().numpy(),
            "b",
        )
        # Shade between the lower and upper confidence bounds
        ax.fill_between(
            test_x_f[_task].detach().cpu().squeeze().numpy(),
            lower.detach().cpu().numpy(),
            upper.detach().cpu().numpy(),
            alpha=0.5,
        )
        #     ax.set_ylim([-3, 3])
        ax.legend(["Training Data", "Test Data", "Mean", "95% Confidence"], fontsize=16)
        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.tick_params(axis="both", which="minor", labelsize=16)
        ax.set_ylabel("Passenger Volume (Normalized)", fontsize=16)
        ax.set_xlabel("Hours (Zoomed to Test)", fontsize=16)
        ax.set_xticks([])

        plt.xlim([1250, 1680])

plt.tight_layout()
plt.show()
