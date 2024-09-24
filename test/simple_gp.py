import math
import torch
import gpytorch
from matplotlib import pyplot as plt
import os
from torch.distributions import Distribution
from torch.utils import benchmark


# smoke_test = "CI" in os.environ
# training_iter = 2 if smoke_test else 50

# # Training data is 100 points in [0,1] inclusive regularly spaced
# train_x = torch.linspace(0, 1, 100)
# # True function is sin(2*pi*x) with Gaussian noise
# train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * math.sqrt(
#     0.04
# )


# # Wrap training, prediction and plotting from the ExactGP-Tutorial into a function,
# # so that we do not have to repeat the code later on
# def train(model, likelihood, training_iter=training_iter):
#     # Use the adam optimizer
#     optimizer = torch.optim.Adam(
#         model.parameters(), lr=0.1
#     )  # Includes GaussianLikelihood parameters

#     # "Loss" for GPs - the marginal log likelihood
#     mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

#     for i in range(training_iter):
#         # Zero gradients from previous iteration
#         optimizer.zero_grad()
#         # Output from model
#         output = model(train_x)
#         # Calc loss and backprop gradients
#         loss = -mll(output, train_y)
#         loss.backward()
#         optimizer.step()


# def predict(model, likelihood, test_x=torch.linspace(0, 1, 51)):
#     model.eval()
#     likelihood.eval()
#     # Make predictions by feeding model through likelihood
#     with torch.no_grad(), gpytorch.settings.fast_pred_var():
#         # Test points are regularly spaced along [0,1]
#         return likelihood(model(test_x))


# def plot(observed_pred, test_x=torch.linspace(0, 1, 51)):
#     with torch.no_grad():
#         # Initialize plot
#         f, ax = plt.subplots(1, 1, figsize=(4, 3))

#         # Get upper and lower confidence bounds
#         lower, upper = observed_pred.confidence_region()
#         # Plot training data as black stars
#         ax.plot(train_x.numpy(), train_y.numpy(), "k*")
#         # Plot predictive means as blue line
#         ax.plot(test_x.numpy(), observed_pred.mean.numpy(), "b")
#         # Shade between the lower and upper confidence bounds
#         ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
#         ax.set_ylim([-3, 3])
#         ax.legend(["Observed Data", "Mean", "Confidence"])
#         return f


# class SimpleSincKernel(gpytorch.kernels.Kernel):
#     has_lengthscale = True

#     # this is the kernel function
#     def forward(self, x1, x2, **params):
#         # apply lengthscale
#         x1_ = x1.div(self.lengthscale)
#         x2_ = x2.div(self.lengthscale)
#         # calculate the distance between inputs
#         diff = self.covar_dist(x1_, x2_, **params)
#         # prevent divide by 0 errors
#         diff.where(diff == 0, torch.as_tensor(1e-20))
#         # return sinc(diff) = sin(diff) / diff
#         return torch.sin(diff).div(diff)


# # Use the simplest form of GP model, exact inference
# class SimpleSincGPModel(gpytorch.models.ExactGP):
#     def __init__(self, train_x, train_y, likelihood):
#         super().__init__(train_x, train_y, likelihood)
#         self.mean_module = gpytorch.means.ConstantMean()
#         self.covar_module = SimpleSincKernel()

#     def forward(self, x):
#         mean_x = self.mean_module(x)
#         covar_x = self.covar_module(x)
#         return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# # We will use the simplest form of GP model, exact inference
# class ExactGPModel(gpytorch.models.ExactGP):
#     def __init__(self, train_x, train_y, likelihood):
#         super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
#         self.mean_module = gpytorch.means.ConstantMean()
#         self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

#     def forward(self, x):
#         mean_x = self.mean_module(x)
#         covar_x = self.covar_module(x)
#         return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# # initialize the new model
# likelihood = gpytorch.likelihoods.GaussianLikelihood()
# model = ExactGPModel(train_x, train_y, likelihood)

# # set to training mode and train
# model.train()
# likelihood.train()
# train(model, likelihood)

# # Get into evaluation (predictive posterior) mode and predict
# model.eval()
# likelihood.eval()
# observed_pred = predict(model, likelihood)
# # plot results
# plot(observed_pred)
# # plt.show()

# print(model.state_dict())
# torch.save(model.state_dict(), "./model/model_state.pth")


# state_dict = torch.load("./model/model_state.pth")
# model = ExactGPModel(train_x, train_y, likelihood)  # Create a new GP model

# model.load_state_dict(state_dict)
# print(model.state_dict())


# # Toy data
# X = torch.randn(10, 1)

# # Base kernels
# rbf_kernel_1 = gpytorch.kernels.RBFKernel()
# cos_kernel_1 = gpytorch.kernels.CosineKernel()
# rbf_kernel_2 = gpytorch.kernels.RBFKernel()
# cos_kernel_2 = gpytorch.kernels.CosineKernel()

# # Implementation 1:
# spectral_mixture_kernel = (rbf_kernel_1 * cos_kernel_1) + (rbf_kernel_2 * cos_kernel_2)
# covar = spectral_mixture_kernel(X)

# # Implementation 2:
# covar = rbf_kernel_1(X) * cos_kernel_1(X) + rbf_kernel_2(X) * cos_kernel_2(X)


# d = 3
# batch_univariate_rbf_kernel = gpytorch.kernels.RBFKernel(
#     batch_shape=torch.Size([d]),  # A batch of d...
#     ard_num_dims=1,  # ...univariate kernels
# )

# n = 10
# X = torch.randn(n, d)  # Some random data in a n x d matrix
# batched_dimensions_of_X = X.mT.unsqueeze(-1)  # Now a d x n x 1 tensor

# univariate_rbf_covars = batch_univariate_rbf_kernel(batched_dimensions_of_X)
# print(univariate_rbf_covars.shape)  # d x n x n

# additive_covar = univariate_rbf_covars.sum(
#     dim=-3
# )  # Computes the sum over the batch dimension
# print(additive_covar.shape)  # n x n


# d = 10
# n = 500
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# print(device)

# X = torch.randn(n, d, device=device)

# naive_additive_kernel = (
#     gpytorch.kernels.RBFKernel(ard_num_dims=1, active_dims=[0])
#     + gpytorch.kernels.RBFKernel(ard_num_dims=1, active_dims=[1])
#     + gpytorch.kernels.RBFKernel(ard_num_dims=1, active_dims=[2])
#     + gpytorch.kernels.RBFKernel(ard_num_dims=1, active_dims=[3])
#     + gpytorch.kernels.RBFKernel(ard_num_dims=1, active_dims=[4])
#     + gpytorch.kernels.RBFKernel(ard_num_dims=1, active_dims=[5])
#     + gpytorch.kernels.RBFKernel(ard_num_dims=1, active_dims=[6])
#     + gpytorch.kernels.RBFKernel(ard_num_dims=1, active_dims=[7])
#     + gpytorch.kernels.RBFKernel(ard_num_dims=1, active_dims=[8])
#     + gpytorch.kernels.RBFKernel(ard_num_dims=1, active_dims=[9])
# ).to(device=device)

# with gpytorch.settings.lazily_evaluate_kernels(False):
#     print(
#         benchmark.Timer(
#             stmt="naive_additive_kernel(X)",
#             globals={"naive_additive_kernel": naive_additive_kernel, "X": X},
#         ).timeit(100)
#     )


# batch_univariate_rbf_kernel = gpytorch.kernels.RBFKernel(
#     batch_shape=torch.Size([d]),
#     ard_num_dims=1,
# ).to(device=device)
# with gpytorch.settings.lazily_evaluate_kernels(False):
#     print(
#         benchmark.Timer(
#             stmt="batch_univariate_rbf_kernel(X.mT.unsqueeze(-1)).sum(dim=-3)",
#             globals={
#                 "batch_univariate_rbf_kernel": batch_univariate_rbf_kernel,
#                 "X": X,
#             },
#         ).timeit(100)
#     )


################ simple GP


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    # the operation that transform the input into output
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# train_x = torch.linspace(0, 1, 100)
# train_y = torch.sin(train_x * 2 * math.pi) + torch.randn(train_x.shape) * 0.2

# likelihood = gpytorch.likelihoods.GaussianLikelihood()
# model = ExactGPModel(train_x, train_y, likelihood)


# # switch to train mode
# model.train()
# likelihood.train()

# # Use the adam optimizer
# optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
# # print(model.parameters)

# # "Loss" for GPs - the marginal log likelihood - log(P(Y|X, theta))
# mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

# training_iter = 50

# for i in range(training_iter):
#     optimizer.zero_grad()

#     output = model(train_x)

#     loss = -mll(output, train_y)
#     loss.backward()

#     print(
#         "Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f"
#         % (
#             i + 1,
#             training_iter,
#             loss.item(),
#             model.covar_module.base_kernel.lengthscale.item(),
#             model.likelihood.noise.item(),
#         )
#     )

#     optimizer.step()


# # Get into evaluation (predictive posterior) mode
# model.eval()
# likelihood.eval()

# test_x = torch.linspace(0, 1, 51)

# with torch.no_grad(), gpytorch.settings.fast_pred_var():
#     observed_pred = likelihood(model(test_x))

# # Initialize plot
# f, ax = plt.subplots(1, 1, figsize=(4, 3))

# # Get upper and lower confidence bounds
# lower, upper = observed_pred.confidence_region()
# # Plot training data as black stars
# ax.plot(train_x.numpy(), train_y.numpy(), "k*")
# # Plot predictive means as blue line
# ax.plot(test_x.numpy(), observed_pred.mean.numpy(), "b")
# # Shade between the lower and upper confidence bounds
# ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
# ax.set_ylim([-3, 3])
# ax.legend(["Observed Data", "Mean", "Confidence"])
# plt.show()


class SpectralMixtureGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(SpectralMixtureGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


train_x = torch.linspace(0, 1, 15)
train_y = torch.sin(train_x * (2 * math.pi))

likelihood = gpytorch.likelihoods.GaussianLikelihood()
# model = SpectralMixtureGPModel(train_x, train_y, likelihood)
model = ExactGPModel(train_x, train_y, likelihood)
print(model.parameters)

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

training_iter = 100
for i in range(training_iter):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    print("Iter %d/%d - Loss: %.3f" % (i + 1, training_iter, loss.item()))
    optimizer.step()


# Test points every 0.1 between 0 and 5
test_x = torch.linspace(0, 5, 51)

# Get into evaluation (predictive posterior) mode
model.eval()
likelihood.eval()

# The gpytorch.settings.fast_pred_var flag activates LOVE (for fast variances)
# See https://arxiv.org/abs/1803.06058
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    # Make predictions
    observed_pred = likelihood(model(test_x))

    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(4, 3))

    # Get upper and lower confidence bounds
    lower, upper = observed_pred.confidence_region()
    # Plot training data as black stars
    ax.plot(train_x.numpy(), train_y.numpy(), "k*")
    # Plot predictive means as blue line
    ax.plot(test_x.numpy(), observed_pred.mean.numpy(), "b")
    # Shade between the lower and upper confidence bounds
    ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    ax.set_ylim([-3, 3])
    ax.legend(["Observed Data", "Mean", "Confidence"])
    plt.show()
