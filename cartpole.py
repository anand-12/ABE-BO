import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from botorch.models import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_mll
from botorch.acquisition import (
    ExpectedImprovement,
    UpperConfidenceBound,
    ProbabilityOfImprovement,
    LogExpectedImprovement
)
from scipy.stats import multivariate_t
from botorch.acquisition.analytic import LogProbabilityOfImprovement
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel, RFFKernel
from botorch.acquisition.analytic import PosteriorMean
from botorch.utils.transforms import normalize, unnormalize, standardize
import warnings
import random
import os
import time
import gpytorch
import gym

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64

def evaluate_policy(policy_params, env, n_episodes=10, max_steps=500):
    total_rewards = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        for _ in range(max_steps):
            action = 1 if np.dot(policy_params, state) > 0 else 0
            state, reward, done, _, _ = env.step(action)
            episode_reward += reward
            if done:
                break
        total_rewards.append(episode_reward)
    return total_rewards

def cartpole_objective(X):
    env = gym.make('CartPole-v1')
    print(f"Evaluating {len(X)} policies")
    all_rewards = []
    for x in X:
        print(f"Evaluating policy: {x}")
        rewards = evaluate_policy(x.cpu().numpy(), env)
        all_rewards.append(rewards)
        print(f"Policy evaluation complete. Mean Reward: {np.mean(rewards):.2f}")
    env.close()
    return torch.tensor([np.mean(r) for r in all_rewards], dtype=dtype, device=device).unsqueeze(-1), all_rewards

class ABEBO:
    def __init__(self, bounds, acquisition_functions):
        self.bounds = bounds
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.double
        self.acquisition_functions = acquisition_functions
        self.use_least_risk = True

        self.d = len(acquisition_functions)
        self.r0 = 0.5 * torch.ones(self.d, dtype=self.dtype, device=self.device)
        self.T0 = 0.25 * torch.eye(self.d, dtype=self.dtype, device=self.device)
        self.kappa0 = 1.0
        self.nu0 = self.d

    def student_t_posterior(self, losses):
        m = len(losses)
        losses_mean = torch.mean(losses)
        
        kappa_m = self.kappa0 + m
        nu_m = self.nu0 + m
        
        r_m = (self.kappa0 * self.r0 + m * losses_mean) / kappa_m
        
        S = torch.cov(losses.T) if m > 1 else torch.var(losses).unsqueeze(0).unsqueeze(0)
        T_m = self.T0 + m * S + (self.kappa0 * m / kappa_m) * torch.outer(losses_mean - self.r0, losses_mean - self.r0)
        
        dof = float(nu_m - self.d + 1)
        loc = r_m
        scale = T_m / (kappa_m * dof)
        
        return dof, loc, scale

    def propose_location(self, model, best_f):
        normalized_bounds = torch.stack([torch.zeros(self.bounds.shape[1]), torch.ones(self.bounds.shape[1])])
        
        candidate_acqs = []
        acq_values = []

        for af in self.acquisition_functions:
            if "UCB" in af:
                acq = UpperConfidenceBound(model=model, beta=float(af.split('_')[1]))
            elif af in ['EI', 'LogEI']:
                acq = ExpectedImprovement(model=model, best_f=best_f) if af == 'EI' else LogExpectedImprovement(model=model, best_f=best_f)
            elif af in ['PI', 'LogPI']:
                acq = ProbabilityOfImprovement(model=model, best_f=best_f) if af == 'PI' else LogProbabilityOfImprovement(model=model, best_f=best_f)
            elif af == 'PM':
                acq = PosteriorMean(model=model)
            else:
                raise ValueError(f"Unsupported acquisition function: {af}")

            candidates, _ = optimize_acqf(
                acq_function=acq,
                bounds=normalized_bounds,
                q=1,
                num_restarts=2,
                raw_samples=20,
            )
            candidate_acqs.append(candidates)
            
            acq_value = acq(candidates)
            
            if af in ['LogEI', 'LogPI']:
                acq_value = torch.exp(acq_value) + 1e-10  
            elif af == 'UCB':
                acq_value = (acq_value - best_f) / (best_f + 1e-6) 
            elif af == 'PM':
                acq_value = (acq_value - model.train_targets.mean()) / (model.train_targets.std() + 1e-6) 
            elif af in ['EI', 'PI']:
                acq_value = (acq_value - 0) / (best_f + 1e-6)
            
            acq_values.append(acq_value)

        acq_values = torch.stack(acq_values)
        acq_values = (acq_values - acq_values.min()) / (acq_values.max() - acq_values.min() + 1e-10)
        
        losses = -acq_values.squeeze()
        
        dof, loc, scale = self.student_t_posterior(losses)
        weights = self.compute_abe_weights(dof, loc, scale)
        
        num_samples = 1000
        least_risk_counts = torch.zeros(len(self.acquisition_functions))
        
        for _ in range(num_samples):
            sample = self.sample_from_posterior(dof, loc, scale)
            least_risk_index = torch.argmin(sample)
            least_risk_counts[least_risk_index] += 1
        
        best_af_index = torch.argmax(least_risk_counts)
        print(f"Chosen acquisition function index: {best_af_index}")
        return candidate_acqs[best_af_index], best_af_index

    def sample_from_posterior(self, dof, loc, scale):
        z = torch.randn_like(loc)
        chi2 = torch.distributions.chi2.Chi2(dof).sample()
        return loc + z * torch.sqrt(torch.diag(scale) * dof / chi2)

    def compute_abe_weights(self, dof, loc, scale):
        n_samples = 1000
        
        loc_np = loc.cpu().detach().numpy()
        scale_np = scale.cpu().detach().numpy()
        
        risk_samples = multivariate_t.rvs(df=dof, loc=loc_np, shape=scale_np, size=n_samples)
        risk_samples = torch.tensor(risk_samples, dtype=self.dtype, device=self.device)
        
        weights = torch.zeros(self.d, dtype=self.dtype, device=self.device)
        for sample in risk_samples:
            best_idx = torch.argmin(sample)
            weights[best_idx] += 1
        
        return weights / n_samples

def get_next_points(train_X, train_Y, best_train_Y, bounds, acq_functions, kernel, n_points=1, gains=None, acq_weight='bandit', use_abe=False, abe_optimizer=None):
    base_kernel = {
        'Matern52': MaternKernel(nu=2.5, ard_num_dims=train_X.shape[-1]),
        'RBF': RBFKernel(ard_num_dims=train_X.shape[-1]),
        'Matern32': MaternKernel(nu=1.5, ard_num_dims=train_X.shape[-1]),
        'RFF': RFFKernel(num_samples=1000, ard_num_dims=train_X.shape[-1])
    }[kernel]

    single_model = SingleTaskGP(train_X, train_Y, covar_module=ScaleKernel(base_kernel))
    mll = ExactMarginalLogLikelihood(single_model.likelihood, single_model)
    with gpytorch.settings.cholesky_jitter(1e-1):
        fit_gpytorch_mll(mll)

    if use_abe:
        if abe_optimizer is None:
            raise ValueError("ABE optimizer should be provided when use_abe is True")
        candidates, chosen_acq_index = abe_optimizer.propose_location(single_model, best_train_Y)
    else:
        acq_function_map = {
            'UCB': lambda beta: UpperConfidenceBound(model=single_model, beta=beta),
            'LogEI': LogExpectedImprovement(model=single_model, best_f=best_train_Y),
            'LogPI': LogProbabilityOfImprovement(model=single_model, best_f=best_train_Y),
            'PM': PosteriorMean(model=single_model)
        }

        candidates_list = []
        for acq_name in acq_functions:
            if acq_name.startswith('UCB'):
                beta = float(acq_name.split('_')[1])
                acq_function = acq_function_map['UCB'](beta)
            elif acq_name in acq_function_map:
                acq_function = acq_function_map[acq_name]
            else:
                raise ValueError(f"Unsupported acquisition function: {acq_name}")

            candidates, _ = optimize_acqf(
                acq_function=acq_function,
                bounds=bounds,
                q=n_points,
                num_restarts=2,
                raw_samples=20,
                options={"batch_limit": 5, "maxiter": 200}
            )
            candidates_list.append(candidates)

        if not candidates_list:
            print("Warning: No valid acquisition functions. Using Expected Improvement.")
            ei = ExpectedImprovement(model=single_model, best_f=best_train_Y)
            candidates, _ = optimize_acqf(
                acq_function=ei,
                bounds=bounds,
                q=n_points,
                num_restarts=2,
                raw_samples=20,
                options={"batch_limit": 5, "maxiter": 200}
            )
            candidates_list = [candidates]
            acq_functions = ['EI']

        if acq_weight == 'random' or gains is None or len(gains) == 0:
            chosen_acq_index = np.random.choice(len(candidates_list))
        else:  # bandit
            eta = 0.1
            logits = np.array(gains[:len(candidates_list)])
            logits -= np.max(logits)
            exp_logits = np.exp(eta * logits)
            probs = exp_logits / np.sum(exp_logits)
            chosen_acq_index = np.random.choice(len(candidates_list), p=probs)

        candidates = candidates_list[chosen_acq_index]

    return candidates, chosen_acq_index, single_model

def bayesian_optimization(args):
    num_iterations = 50
    initial_points = 5
    env = gym.make('CartPole-v1')
    obs_dim = env.observation_space.shape[0]
    bounds = torch.tensor([[-1.0] * obs_dim, [1.0] * obs_dim], dtype=dtype, device=device)

    # Draw initial points
    train_X = draw_sobol_samples(bounds=bounds, n=initial_points, q=1).squeeze(1)
    train_Y, all_rewards = cartpole_objective(train_X)

    best_train_Y = train_Y.max().item()

    gains = np.zeros(len(args.acquisition))
    max_values = [best_train_Y]
    all_episodes_rewards = all_rewards
    chosen_acq_functions = []
    kernel_names = []

    abe_optimizer = ABEBO(bounds, args.acquisition) if args.use_abe else None

    for i in range(num_iterations):
        print(f"Running iteration {i+1}/{num_iterations}, Best value so far = {best_train_Y:.4f}")

        fit_bounds = torch.stack([torch.min(train_X, 0)[0], torch.max(train_X, 0)[0]])

        train_X_normalized = normalize(train_X, bounds=fit_bounds)
        train_Y_standardized = standardize(train_Y)

        best_f = (best_train_Y - train_Y.mean()) / train_Y.std()

        new_candidates_normalized, chosen_acq_index, model = get_next_points(
            train_X_normalized, train_Y_standardized, 
            best_f, normalize(bounds, fit_bounds),
            args.acquisition, args.kernel, 1, gains, args.acq_weight, args.use_abe, abe_optimizer
        )

        # Unnormalize the candidates
        new_candidates = unnormalize(new_candidates_normalized, bounds=fit_bounds)
        new_Y, new_rewards = cartpole_objective(new_candidates)

        train_X = torch.cat([train_X, new_candidates])
        train_Y = torch.cat([train_Y, new_Y])

        best_train_Y = train_Y.max().item()

        max_values.append(best_train_Y)
        all_episodes_rewards.extend(new_rewards)
        chosen_acq_functions.append(args.acquisition[chosen_acq_index])
        kernel_names.append(args.kernel)

        posterior_mean = model.posterior(new_candidates_normalized).mean
        reward = posterior_mean.mean().item()
        gains[chosen_acq_index] += reward

    return max_values, all_episodes_rewards, chosen_acq_functions, kernel_names

def run_experiments(args):
    all_results = []

    for seed in range(args.seed, args.seed+args.experiments):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        start_time = time.time()
        max_values, all_episodes_rewards, chosen_acq_functions, kernel_names = bayesian_optimization(args)
        end_time = time.time()

        experiment_time = end_time - start_time
        all_results.append([max_values, all_episodes_rewards, experiment_time, chosen_acq_functions, kernel_names])
        print(f"Experiment {seed} completed in {experiment_time:.2f} seconds")

    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BoTorch Bayesian Optimization for CartPole')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--acquisition', nargs='+', default=['LogEI', 'LogPI', 'UCB_0.1', 'UCB_0.3', 'UCB_0.7', 'UCB_0.9'], 
                        help='List of acquisition functions to use. For UCB, use format UCB_beta (e.g., UCB_0.1)')
    parser.add_argument('--kernel', type=str, default='Matern52', 
                        choices=['Matern52', 'RBF', 'Matern32', 'RFF'],
                        help='GP kernel to use')
    parser.add_argument('--experiments', type=int, default=1, help='Number of experiments to run')
    parser.add_argument('--acq_weight', type=str, default='bandit', choices=['random', 'bandit'],
                        help="Method for selecting acquisition function: random or bandit")
    parser.add_argument('--use_abe', action='store_true', help='Use Improved Approximate Bayesian Ensembles with least risk strategy')
    args = parser.parse_args()

    print("Acquisition functions:", args.acquisition)

    all_results = run_experiments(args)

    # Save results
    all_results_np = np.array(all_results, dtype=object)
    os.makedirs("./Results_CartPole", exist_ok=True)
    
    if args.use_abe:
        filename = f"GPHedge_abe_least_risk"
    else:
        filename = f"GPHedge_{args.acq_weight}"
    np.save(f"./Results_CartPole/CartPole_{filename}.npy", all_results_np)

    print(f"Results saved to ./Results_CartPole/CartPole_{filename}.npy")
    print("Results include rewards for all episodes in each iteration.")
