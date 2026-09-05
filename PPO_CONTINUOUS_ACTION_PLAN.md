# Continuous Box PPO and gSDE implementation plan

Prepared for Sol on 2026-09-05. This is an implementation handoff, not implemented functionality or a training result.

Repository inspected at `4c954f3e98309b85e12dcd2527053bb21f97a23e`; the existing `.gitignore` modification is unrelated and must be preserved. The local manifest records Julia 1.12.6 and Flux 0.16.10. Recheck the working tree before implementation. No applicable repository `AGENTS.md` was found during this review.

## 1. Recommended outcome and scope

Extend the existing `ppo` entry point to support explicitly declared, finite continuous boxes while retaining discrete PPO. Implement a diagonal Gaussian baseline first, then an optional gSDE actor using the same rollout and optimization code. Make gSDE the recommended experiment preset, not a silent replacement for all continuous policies.

The end-to-end pipeline should be:

1. Encode observations into an `obs_dim × num_envs` Float32 matrix.
2. Compute a mean action and exploration scale for each observation.
3. Sample an unconstrained policy action; preserve that sample and its old log probability.
4. Map it into the declared box and decode it into the MDP's native action type.
5. Step environments, retain final observations before resets, and compute correct bootstrap targets.
6. Reevaluate stored policy actions under the updated actor during PPO minibatches.
7. Evaluate the trained policy with deterministic actions, frozen normalization, and original rewards.

Implement CPU vectorized training first. Do not add GPU support, SAC, a general distribution framework, full-covariance policies, recurrent policies, or a Python environment bridge as prerequisites. A scalar or shaped native MDP action should be supported through a decoder, while neural-network actions remain flattened internally. Reject infinite bounds and zero-width dimensions initially with clear errors.

gSDE is attractive when independent action noise produces unhelpful rapid changes. Its benefit here remains an empirical question: measure return per environment step, return per second, and action smoothness against independent Gaussian exploration.

## 2. What was checked in Stable-Baselines3

The linked `master` sources were inspected through web retrieval. These are moving references, not a pinned checkout; shell access could not resolve GitHub to obtain a commit SHA. Before producing parity fixtures, record the exact installed SB3 version or source commit. Do not assume every historical SB3 implementation behaves like the inspected files.

| Source | Relevant finding |
| --- | --- |
| [Distributions](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/distributions.py), `DiagGaussianDistribution`, `StateDependentNoiseDistribution` | Box policies select Gaussian or gSDE distributions. gSDE samples feature-to-action noise weights, obtains variance from squared features and weight scales, and sums likelihoods over action coordinates. `full_std` specifies separate feature/action scales; it does not mean full action covariance. |
| [ActorCriticPolicy](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/policies.py) | Actor defaults include two 64-unit tanh layers, `log_std_init=0`, `full_std=true`, `use_expln=false`, and `squash_output=false`. The gSDE variance path detaches features. Mean-path features still learn. Adam uses epsilon `1e-5`. Squashing in this policy class requires gSDE. |
| [On-policy collection](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/on_policy_algorithm.py) | Noise is refreshed at rollout start and at a positive `sde_sample_freq`. The collector executes bounded actions but stores policy actions. Timeouts bootstrap from terminal observations. |
| [PPO training](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ppo/ppo.py) | Training reevaluates stored actions, uses scalar action log probabilities, supports approximate-KL early stopping, and reports clipping diagnostics. The inspected training loop does not resample exploration weights for minibatch likelihood evaluation. |
| [gSDE tests](https://github.com/DLR-RM/stable-baselines3/blob/master/tests/test_sde.py) | Useful reference tests cover scale gradients, invalid discrete/gSDE combinations, and squashing/action scaling. Add stronger explicit out-of-range fixtures locally. |
| [VecNormalize](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/vec_env/vec_normalize.py) | Observation statistics and discounted-return statistics are separate; reward normalization divides by return standard deviation. Terminal observations are normalized, evaluation statistics can be frozen, and statistics can be saved. |
| [RL Zoo PPO presets](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml) | The inspected Pendulum preset uses gSDE with resampling every four steps and zero entropy coefficient. Other tasks use different schedules and scales: these are task-specific examples. |
| [Original gSDE paper](https://arxiv.org/abs/2005.05719) | Motivation: temporally structured, state-dependent exploration for continuous control. It does not establish that gSDE is best for this repository's environments. |

The detailed interfaces, rollout layout, numerical choices, and acceptance gates below are proposed for CleanRL.jl. They are not a request to port SB3's class hierarchy or claim numerical identity with SB3.

## 3. Repository-specific changes that are necessary

| Current location | Problem | Proposed change |
| --- | --- | --- |
| `src/utils/pomdp_env.jl`, `MDPEnv` | Constructor defaults eagerly collect `POMDPs.actions(mdp)`; actions are stored as `Vector{A}`; default decoding uses an integer index. Continuous metadata is absent. | Add explicit discrete/box metadata and defer enumeration until the discrete branch is selected. A Box constructor must never enumerate the MDP's actions. |
| Same file, `action_dim` | Infers dimension by sampling, which consumes RNG state and does not provide bounds. | Read box dimension and shape from metadata; retain compatibility where needed for existing DDPG callers. |
| Same file, `step!` | A single terminal flag combines true termination and `max_steps`. | Expose true termination and truncation separately while retaining the legacy done accessor for other algorithms. |
| `src/utils/multi_thread_env.jl` | Matrix action dispatch already selects one column per environment, but metadata validation and distinct ending flags are missing. Accessors return reused mutable buffers. | Preserve column dispatch; validate all action specs and observation sizes at construction; expose/copy ending masks deliberately. |
| `src/algorithms/ppo.jl`, actor construction and helpers | Actor is assumed to be a categorical `Chain`; action counts, integer buffers, and `vec(mb_actions)` assume discrete actions. | Dispatch policy sampling/evaluation by actor type; keep continuous minibatch actions as matrices. |
| Same file, collection/bootstrap | After the last transition, environments may be reset before `is_terminated(env)` is queried again for GAE. This can erase the final transition's done flag and bootstrap across episodes. `next_done` can also alias the mutable environment mask. | Use transition-indexed ending flags, capture final observations before resets, and test the rollout-ending boundary explicitly. |
| `src/utils/replay_buffer.jl` | Buffer already allocates from prototype shapes and appends time as the final dimension. | Reuse it with continuous prototypes; verify layout and full-rollout overwrite ordering. Do not redesign replay storage unnecessarily. |
| `src/utils/networks.jl` | Existing independent actor/critic trunks and orthogonal output gains are reusable. | Add a continuous actor with separate feature trunk, mean head, and trainable noise parameters. |
| PPO configuration/minibatches | Integer division can create invalid minibatches or silently train zero updates; advantage normalization is unsafe for size one. | Validate before allocating networks or creating loggers. |
| `test/runtests.jl` | Only discrete environment/PPO smoke tests and standalone GAE tests exist. | Add continuous numerical, rollout-boundary, gradient, and learning tests. |

Keep DQN/A2C discrete. Do not refactor DDPG's noisy action head into PPO: it neither provides a policy likelihood nor handles arbitrary bounds. Preserve its existing adapter signatures unless a focused compatibility adjustment is necessary.

## 4. Public contracts and file layout

### Action-space metadata

Add `src/utils/action_spaces.jl`, included before `pomdp_env.jl`, containing small concrete descriptors such as `DiscreteActionSpec` and `BoxActionSpec`. A Box stores flattened Float32 `low`, `high`, and its original shape. Validate matching shapes, nonempty dimension, finite representable bounds, strictly positive widths, and finite representable midpoint/scale calculations.

Proposed usage, to implement and document:

```julia
spec = BoxActionSpec(Float32[-2, 0], Float32[2, 3])
factory = () -> MDPEnv(mdp;
    action_space=spec,
    action_decoder=a -> MyAction(a[1], a[2]),
    max_steps=200,
)
actor, critic = ppo(factory, config)
```

For scalar actions document `action_decoder=only`; for matrix-shaped actions decode with the stored shape. Decoder inputs are already in environment units, so do not scale them again. Supply an RNG-aware uniform box random sampler by default. Use explicit metadata rather than guessing from an `actions(mdp)` container.

Preserve `MDPEnv(mdp)` and legacy discrete/custom keyword behavior. Distinguish omitted defaults from explicit `actions`, decoder, or sampler arguments so the Box branch can choose appropriate defaults without evaluating `collect(...)`. Error on conflicting declarations. Keep `action_count` meaningful for discrete spaces and provide an actionable error for Box callers. Add `action_space(env)` and `single_action_space(env)` helpers.

### Policy and exploration state

Add `src/utils/continuous_policies.jl` with two concrete policy types, e.g. `DiagGaussianActor` and `GSDEActor`. Retain existing categorical helpers or give them thin adapters rather than rewriting their mathematics.

Proposed internal interfaces:

```julia
sample_policy(rng, actor, obs_matrix, exploration_state)
# -> (raw_action, logprob), shapes (action_dim, batch), (batch,)

evaluate_policy(actor, obs_matrix, raw_actions)
# -> (logprob, entropy), each (batch,) for the unsquashed modes

reset_noise!(rng, exploration_state, actor; env_indices=:all)
predict(actor, obs; deterministic=true, exploration_state=nothing, rng=...)
# -> bounded environment-space action(s), with documented single/batch shape
```

Make the actor retain action-bound/normalization metadata needed for deployment, while Flux trains only trunk, mean head, and scale parameters. Keep sampled noise matrices, RNGs, buffers, and running statistics outside the optimizer tree. Use explicit trainable fields supported by the installed Flux version and test their enumeration. Return `(actor, critic)` as today; do not leave a trained actor unusable without hidden environment state. A callable continuous actor may delegate to deterministic `predict`; document this explicitly.

Preserve discrete configuration defaults. Add `use_sde::Bool=false`, `sde_sample_freq::Int=-1`, `log_std_init::Float32=0f0`, `full_std::Bool=true`, `action_transform::Symbol=:clip`, `target_kl` (disabled by default), `seed`, and a configurable gradient norm bound. `:tanh` is a later phase. Initially use exponential scales; defer `use_expln` and learned separate exploration features until a measured need appears. Validate or extend `ConfigParser.argparse_struct` for Symbol/optional fields and Boolean options; avoid making CLI parsing an untested side effect.

Define `use_sde=true` on discrete actions as an error. Accept only `sde_sample_freq=-1` or a positive integer; define `-1` as rollout-start-only refresh and `1` as every environment step. Other values should fail early.

## 5. Distribution mathematics and action bounds

All equations below use Julia's batch-last layout. Let `A` be flattened action dimension, `F` actor feature dimension, and `B` batch size. `H = trunk(obs)` has shape `F × B`; `μ = mean_head(H)` has shape `A × B`.

### Independent Gaussian baseline

Train `log_std` with shape `A × 1`; broadcast `σ = exp(log_std)` across observations. Sample `u = μ + σ .* ξ`, with independent standard normal `ξ` for every action coordinate and environment at every step.

For both policy types, let `v` be the marginal action variance. Compute one log probability per column:

```text
logp[b] = -0.5 * sum_a(log(2π) + log(v[a,b])
                      + (u[a,b] - μ[a,b])² / v[a,b])
entropy[b] = 0.5 * sum_a(log(2πe) + log(v[a,b]))
```

Use Float32 constants in hot paths and sum along action dimension only. Gradient evaluation treats stored actions and old log probabilities as constants. Do not resample actions inside the PPO surrogate loss.

### gSDE actor

Train `L` with shape `F × A`; set weight scales `S = exp.(L)`. For each environment `e`, sample independent standard normals `Z_e` of shape `F × A` and cache `W_e = S .* Z_e`. Then:

```text
u[:,e] = μ[:,e] + transpose(W_e) * H[:,e]
v = transpose(S.^2) * (H.^2)           # A × B
```

With `full_std=false`, train `F × 1` scales and broadcast them across action columns, but keep sampled weights independent across columns. Sharing scales must not accidentally share action noise.

Use a stopped-gradient copy of `H` on the variance/noise branch; the mean branch remains differentiable through the trunk. Recompute current features for every likelihood evaluation. Never store old features as substitutes for current ones.

For externally fixed features, these equations give the Gaussian marginal over freshly sampled weights. Holding weights across transitions creates temporal dependence, and later states depend on earlier noise. The PPO marginal likelihood is the practical gSDE surrogate; do not call it an exact joint trajectory likelihood or an exact conditional density given the entire noise-dependent history.

Numerical floor decision: the inspected SB3 density adds epsilon to variance without adding the corresponding noise in its sampler. For a coherent local marginal at fixed features, use `v + εvar` in the density and add an independent `sqrt(εvar) * ξ` residual to sampling, with `εvar=1f-6` initially in normalized policy units. This is an explicit small deviation from SB3 sampling and introduces tiny stepwise noise. Test it and log the floor. For an SB3 sampler parity fixture, compare the structured component separately. Do not introduce a density-only variance clamp while claiming an exactly matching sampler.

At zero features, the residual preserves a defined density and tiny exploration, but learned gSDE scale gradients through those features vanish. Log feature norms and effective action standard deviations. If a task starts/stays near zero features and exploration stalls, first compare the Gaussian baseline; a constant exploration feature is a later, documented ablation.

Refresh all environments' weight matrices at rollout start using current learned scales. For positive frequency `k`, refresh before steps `1, 1+k, 1+2k, ...`, without redundant double sampling at step one. Initially keep matrices across episode resets until the scheduled refresh, matching the inspected collector's schedule; document that choice. No weight sampling is required during `evaluate_policy`. Resample when starting a new stochastic evaluation rollout; deterministic prediction consumes no exploration RNG. Require explicit handling when stochastic evaluation batch size changes, rather than silently sharing one matrix.

### Initial bounds implementation: normalized clipping

Use normalized policy coordinates consistently for both Gaussian and gSDE:

```text
c = (high + low) / 2
d = (high - low) / 2
a_env = c + d .* clamp.(u, -1, 1)
```

Compute midpoint/scale safely at construction. Store the original `u` and `logp(u)` in the rollout. Never overwrite `u` with its clipped version or evaluate an ordinary Gaussian density at the clipped action. Clipping induces boundary probability mass; the stored likelihood is for the latent policy action. PPO can optimize this augmented action representation with a fixed environment action mapping.

This normalized coordinate choice differs from SB3's default unsquashed environment-unit Gaussian. It makes scale settings comparable across heterogeneous bounds. For full training comparisons against SB3, give SB3 the identical normalized action wrapper; otherwise label the comparison unmatched.

Log the fraction of coordinates outside `[-1,1]`, the fraction of transitions with any clipped coordinate, and deterministic mean saturation. Frequent clipping can remove useful distinctions between sampled actions even when likelihood bookkeeping is correct.

### Follow-up bounds implementation: tanh

After clipped-mode correctness and learning gates pass, support `a_env = c + d .* tanh.(u)` for both continuous actors. Supporting it for the independent Gaussian is an intentional extension beyond the inspected SB3 ActorCriticPolicy restriction.

Continue storing `u`, avoiding inverse tanh at numerically saturated actions. When exposing environment-action log density, subtract both Jacobians:

```text
logp_env = logp_u - sum_a(log(d[a]) + log(1 - tanh(u[a])²))
log(1 - tanh(u)²) = 2 * (log(2) - u - softplus(-2u))
```

Use the stable expression. PPO ratios may use `logp_u` in both old and new terms because the fixed transform cancels. Label that convention; never mix raw and transformed log probabilities. Deterministic prediction applies tanh to the mean, without claiming that this is the transformed distribution's mode.

Base-Gaussian entropy is not tanh-action entropy. Initially require `ent_coeff=0` for tanh mode and report base entropy only under an explicit name. A later nonzero transformed-entropy bonus needs fresh current-policy Monte Carlo samples and a tested gradient estimator. Negative current log probability of old rollout actions is a cross-entropy estimate, not generally current-policy entropy.

## 6. Rollout storage and episode boundaries

Use these shapes, with `N=num_envs`, `T=num_steps`, `B=N*T`:

| Field | Collection shape | Flattened training shape |
| --- | --- | --- |
| Encoded/normalized observation | `O × N × T` | `O × B` |
| Raw continuous action | `A × N × T` | `A × B` |
| Discrete action | `N × T` | length `B` |
| Old log probability, value, reward, advantage, return | `N × T` | length `B` |
| True termination, truncation | `N × T` | not needed in actor minibatches |

Column-major reshape puts environments within time steps; use identical ordering for every field. Exercise `A>1`, `N>1`, `T>1` with uniquely tagged fixtures. Only vectorize discrete minibatch actions. Insert `vec(critic(obs))` into value storage explicitly.

For each transition, capture `terminated`, `truncated`, and the successor observation before resetting any environment. Preserve `is_terminated` as legacy episode-done semantics if required by DQN/A2C/DDPG; add clearly named true-termination/truncation accessors for PPO. An environment that reaches a true terminal and time limit together is terminal for bootstrapping.

Use transition-indexed GAE:

```text
δ_t = r_t + γ * (1 - terminated_t) * V(final_or_next_obs_t) - V(obs_t)
A_t = δ_t + γλ * (1 - (terminated_t || truncated_t)) * A_(t+1)
```

At a time limit, bootstrap from the final observation but stop the advantage recursion across reset. At a true terminal, do neither. At a rollout boundary with no episode ending, bootstrap from the continuing observation and initialize the unavailable future advantage to zero.

For efficiency, derive ordinary successor values from the next stored value; batch-evaluate only timeout final observations and continuing rollout-end observations. A straightforward explicit next-value array is acceptable first. Normalize final observations with the same preprocessing used for value inference. Do not also add timeout bootstrap into rewards if using the formula above; that would double count it.

Copy observations and masks into owned rollout storage before calling accessors that mutate reused batch buffers. Replace `deepcopy` with preallocated copies only after aliasing regressions pass. Keep the existing public `gae` helper and its tests, or provide a compatibility overload rather than silently changing its indexing convention.

## 7. PPO optimization, normalization, and performance

### Correctness and stability

Validate positive environment/step/epoch/minibatch counts, `B % num_minibatches == 0`, and minibatch size greater than one when normalizing advantages. Require at least one complete rollout; document rounded-down total steps when the requested budget is not divisible by `B`. Validate finite positive learning rates/scales and valid discount/GAE ranges.

Keep gradient clipping, but verify it covers actor scales as well as actor/critic weights. Initialize separate 64×64 tanh trunks using current network helpers, mean head gain `0.01`, critic gain `1`. Set Adam epsilon explicitly if following the reference policy initialization; verify the installed Optimisers API rather than guessing its keyword names.

Add approximate KL `mean(expm1.(new_logp-old_logp) - (new_logp-old_logp))` and stop remaining minibatches/epochs when it exceeds `1.5 * target_kl`, before applying that minibatch's update. It is a sampled diagnostic, not a hard trust-region guarantee. At unchanged parameters the initial ratios must be one within Float32 tolerance.

The current `[-20,20]` log-ratio clamp changes the surrogate and can conceal collapse. Remove it as ordinary behavior once finite-scale tests pass. Detect nonfinite distributions, likelihoods, gradients, and losses explicitly; reject an unsafe update and return an actionable error with diagnostics. Compute KL from the unclamped log-ratio. If a guarded exponentiation path is needed for overflow, do not continue training on silently altered ratios.

Use `ent_coeff=0` and `clip_value_loss=false` in the continuous preset; retain legacy defaults. If value clipping is later enabled, add a distinct value clip scale, because reward units affect its meaning. The local loss includes a `0.5` factor before multiplying by `v_coef`; SB3's inspected MSE does not. Match effective critic weighting, not just the printed `v_coef`, in parity experiments.

### Normalization

Implement optional PPO-owned running observation statistics and optional reward normalization, keeping statistical accumulation in Float64 and network tensors in Float32. Store the exact normalized observations used during collection; do not renormalize old samples with later statistics. Freeze statistics throughout optimizer epochs.

For reward normalization, maintain per-environment discounted returns, update running return variance, divide rewards by its standard deviation, and reset the return accumulator at episode endings. Record raw rewards separately for episode/evaluation reporting. Make normalization and clipping settings explicit, save their state with the policy, and use frozen observation statistics and original rewards in evaluation. Test terminal-observation preprocessing and loading a saved policy. Normalization can be disabled for the initial well-scaled toy tests.

### CPU efficiency

Keep sampling, actor/critic inference, and likelihood evaluation batched. Compute the gSDE trunk once per actor call and reuse it for mean/variance. Preallocate `F × A × N` exploration matrices and action scratch space; sample with the supplied RNG in place outside AD. Compute variance with matrix multiplication; use per-environment `mul!` or a measured batched kernel for noise. Avoid creating a distribution object per state or coordinate.

Use function boundaries specialized on concrete actor/action types, not runtime Symbol branches inside differentiated array loops. Do not put mutable RNG or environment operations inside `Flux.withgradient`.

The current environment factory may reuse one MDP object. Require that such a model is immutable/thread-safe, or build independent mutable models. Assign independent reproducible environment RNG streams and a separate policy/shuffle stream; avoid sharing one mutable RNG across threads. Preserve the zero-argument factory API and add/document a seeded factory path if needed.

Profile serial versus threaded environment stepping: tiny MDPs can lose time to thread scheduling. Measure BLAS thread counts alongside Julia environment threads to avoid oversubscription. Aggregate training logs once per update; avoid per-minibatch disk logging becoming the bottleneck.

Report compilation/startup separately from warmed collection time, gradient-update time, allocations, and total wall-clock time. Throughput comparisons must use identical network sizes, rollout budgets, thread settings, and logging frequency. Do not assume GPU acceleration or gSDE improves wall time.

## 8. Initial experiment settings

Provide a helper or runnable example constructing a continuous preset rather than changing global defaults:

| Setting | Proposed starting value |
| --- | --- |
| Policy | gSDE, with a matched independent-Gaussian switch |
| Hidden layers | separate actor/critic `[64, 64]`, tanh |
| `num_envs`, `num_steps` | 8, 256: 2,048 transitions/update |
| `num_minibatches`, `update_epochs` | 8, 10: 256 samples/minibatch |
| Learning rate | `3f-4`, annealed |
| Gamma, GAE lambda | `0.99f0`, `0.95f0` |
| Policy clip, gradient norm | `0.2f0`, `0.5f0` |
| Entropy coefficient | `0f0` |
| Value coefficient/clipping | existing local `v_coef=0.5f0`, clipping disabled; note effective weighting above |
| Target KL | `0.02f0` |
| Action transform | normalized `:clip`, later compare `:tanh` |
| gSDE scale initialization | `log_std_init=-2f0`, `full_std=true` |
| gSDE resampling | every 4 vector-environment steps |
| Normalization | observations on for general tasks; rewards opt-in; off for well-scaled numerical fixtures |

These are proposed starting values, not tuned or identical SB3 defaults. A gSDE weight standard deviation is not an action standard deviation: effective scale depends on feature norms. For a fair Gaussian comparison, estimate initial marginal action variances over the same calibration observations and match them approximately, then record both initializations. Do not blindly assign the same `log_std_init` to both actors and call their exploration matched.

After smoke tests, vary only `sde_sample_freq ∈ {1,4,16,-1}` on one control task, select on designated development seeds, and confirm on held-out seeds. Keep total environment interactions and evaluation intervals fixed. Report actual optimizer work when KL stopping makes it differ. Relate resampling to simulator step duration and action repeat, since four steps can mean very different physical intervals.

## 9. Tests and acceptance gates

### Fast deterministic/numerical tests

1. **Box adapter:** asymmetric nonunit bounds, scalar/vector/shaped decoding, malformed bounds, metadata disagreement across environments, and a continuous MDP whose `actions` method deliberately throws if enumerated. Verify construction succeeds through the Box branch and all executed actions are legal.
2. **Gaussian likelihood:** compare batched results to sums of scalar `Distributions.Normal` log densities and entropy. Cover `A=1`, `A>1`, `B=1`, and shuffled minibatches. No accidental reduction over batch.
3. **Gradients/trainability:** actor mean, trunk, and log scales receive finite expected gradients and change after an update. Bounds, RNG state, and noise matrices are not optimizer parameters. Finite-difference scale gradients at fixed observations/actions; isolate the variance branch to test stopped feature gradients while preserving mean-path gradients.
4. **gSDE statistics:** at fixed nonzero features, repeatedly refresh noise and compare empirical mean/variance to the implemented marginal including the residual floor. Check independent environments and action coordinates. With cached weights, verify identical structured noise at identical features and measured cross-state covariance `sum_f H1[f]*H2[f]*S[f,a]^2`; test the small residual separately.
5. **Noise lifecycle:** exact refresh counts for `-1`, `1`, `4`; no refresh during likelihood evaluation; correct use of updated scales at the next rollout; deterministic evaluation consumes no noise; changing evaluation batch size does not silently share matrices.
6. **Storage/bounds:** force raw actions far outside the box and assert stored samples/log probabilities remain raw while executed actions are bounded. Tag all action/environment/time coordinates to verify flattening order. Test buffer rollover over multiple full PPO updates.
7. **GAE integration:** true terminal, timeout, continuing rollout end, and simultaneous terminal/time limit. Put each ending on both an interior and final rollout step; make reset and final observations have different known values. For example, `r=2`, `γ=0.5`, `V(s)=1`, `V(final)=4` gives TD residual `1` at a terminal and `3` at a timeout, with recursion stopped in both cases. Include multiple environments ending at different steps and mutable-buffer alias checks.
8. **PPO invariants:** unchanged-policy ratios approximately one, hand-computed clipped surrogate, KL stopping prevents the triggering update and all remaining epochs, zero/singleton/nondivisible minibatch errors, and finite full updates for both actors.
9. **Normalization/inference:** old stored observations do not change after statistics update, frozen evaluation does not modify statistics, final observations use correct scaling, original reward reporting survives normalization, and a saved/reloaded deployment bundle predicts the same actions.
10. **Tanh follow-up:** stable Jacobian at large positive/negative `u`, asymmetric affine scales, old/new ratio cancellation, finite saturated-action handling, and explicit rejection of unsupported nonzero entropy bonus.

Make numeric fixtures deterministic. Use justified sample-count-based tolerances for statistical tests, not brittle exact sample equality between Julia and PyTorch. For reference fixtures, supply identical means/features/scales/actions to a pinned SB3 version and compare density/entropy/gradients, transposing layouts explicitly and accounting for the documented floor convention. SB3 is a development oracle, not a runtime package dependency.

### Integration and learning checks

Retain all existing tests. Add small bounded-action POMDP fixtures without new simulator dependencies:

- A contextual quadratic-reward one-step task with a known interior optimum in an asymmetric multi-action box. Both actors must improve deterministic evaluation reward over initialization across fixed seeds.
- A simple controlled dynamical system with a time limit, nonzero target, bounded actions, and an action-change cost. Use it to exercise bootstrapping and report exploration smoothness; do not assume it proves general gSDE superiority.

Keep short finite-update smoke tests in routine CI. Put longer learning checks in a separately invoked script with fixed budgets and numerical success thresholds declared before running; derive thresholds from the known task optimum and document any tuning. Then add one Pendulum-like native MDP example with fully specified dynamics, rewards, termination, and action units. Do not label it Gymnasium Pendulum parity unless those details match.

Run the supported package test workflow in a resolved environment containing `test/Project.toml` dependencies. Check whether `Pkg.test()` resolves that test environment correctly before assuming the command works. Record exact commands, Julia/Flux versions, thread counts, seeds, and outcomes. If package/precompile setup blocks tests, report the blocker separately from code correctness.

For the practical benchmark, use at least five seeds and fixed independent evaluation seeds, evaluating e.g. 10 episodes every fixed transition interval. Compare Gaussian versus gSDE under the same transform, preprocessing, architecture, and training budget. Report mean/spread of returns, learning-curve area, time to a declared return threshold, warmed steps/sec, clipping rate, marginal action scales, and within-episode action differences in physical units. Exclude differences across resets from smoothness statistics. Smooth but unproductive behavior is not success.

## 10. Implementation sequence and handoff checklist

1. **Environment foundation:** add action descriptors and separate ending flags, preserve compatibility, add adapter and boundary tests. Fix the existing rollout-end bootstrap issue before attributing learning changes to exploration.
2. **Gaussian PPO:** add the continuous actor/interface, normalized clipping, matrix action storage, deterministic inference, configuration validation, likelihood/gradient tests, and a finite-update smoke test. Run the existing discrete suite.
3. **gSDE:** implement weight-scale parameters, cached per-environment noise, detached variance features, residual-floor convention, refresh schedule, and statistical/gradient tests. Reuse the Gaussian log-density primitive and unchanged PPO loss.
4. **Training support:** add normalization with deployment state, KL stopping/diagnostics, reproducible RNG streams, aggregate logging, and the preset. Verify both toy tasks and profile warmed CPU execution.
5. **Bounded transform follow-up:** add tanh and its dedicated tests. Keep Gaussian/gSDE comparisons under the same transform.
6. **Evidence and documentation:** add the native continuous example, benchmark script/results, and README usage. Record remaining limits and actual verification outcomes; do not claim gSDE improves efficiency without measured evidence.

Expected touched files: new action-space/policy/normalization utilities as needed; `src/CleanRL.jl` include/export wiring; `src/utils/pomdp_env.jl`; `src/utils/multi_thread_env.jl`; focused additions to `src/utils/networks.jl`; `src/algorithms/ppo.jl`; test files included by `test/runtests.jl`; README; and a new `examples/` or `experiments/` runner. Change replay/config utilities only where tests establish a need. Avoid unrelated cleanup and preserve user modifications.

Completion means discrete PPO still passes, finite Box actions work end to end for both exploration modes, likelihood and ending semantics are tested, trained policies retain necessary preprocessing/bounds for evaluation, and efficiency claims are backed by reproducible measurements. If a later phase cannot be completed, report it explicitly rather than presenting Gaussian support alone as the full gSDE handoff implemented.
