# CleanRL.jl
Simple single file implementations of Reinforcement Learning algorithms in Julia.  
Inspired by [CleanRL](https://github.com/vwxyzjn/cleanrl)

## POMDPs.jl MDPs
Algorithms now take a `POMDPs.MDP` or `MDPEnv` instead of constructing hardcoded
`ReinforcementLearning.jl` environments internally:

```julia
using CleanRL
using POMDPs

dqn(mdp, DQNConfig(total_timesteps=100_000))
ppo(mdp, PPOConfig(num_envs=8))
```

`MDPEnv` uses `POMDPs.convert_s(Vector{Float32}, s, mdp)` to turn model states
into neural-network observations. Define that method for struct states:

```julia
POMDPs.convert_s(::Type{Vector{Float32}}, s::MyState, ::MyMDP) = Float32[
  s.x,
  s.y,
  s.v,
]
```

Discrete algorithms enumerate `actions(mdp)` and internally use integer action
indices. PPO also supports explicitly bounded continuous actions without
enumerating `actions(mdp)`:

```julia
spec = BoxActionSpec(Float32[-2, 0], Float32[2, 3])
factory = (index, rng) -> MDPEnv(mdp;
  rng=rng,
  action_space=spec,
  action_decoder=a -> MyAction(a[1], a[2]),
  max_steps=200,
)

# Independent diagonal Gaussian exploration
actor, critic = ppo(factory, PPOConfig(ent_coeff=0f0))

# Generalized state-dependent exploration (recommended experiment preset)
actor, critic = ppo(factory, continuous_ppo_config())
action = predict(actor, observation; deterministic=true)
```

PPO passes an independent seeded RNG to factories accepting `(index, rng)` (or
just `rng`); zero-argument factories remain supported for compatibility. The
policy/shuffle RNG is separate and all streams are derived from `PPOConfig.seed`.

Neural-network actions are flattened and normalized internally. `predict`
returns bounded actions in environment units. For a scalar Box, pass
`action_decoder=only`; matrix-shaped Boxes use their original shape by default.
The initial implementation accepts only finite, nonzero-width bounds. `:clip`
stores and optimizes the unclipped latent action; `:tanh` is also available but
currently requires `ent_coeff=0` because the reported entropy is base-Gaussian
entropy, not transformed-action entropy.

gSDE noise is refreshed at each rollout start and, when `sde_sample_freq=k>0`,
before steps `1+k, 1+2k, ...`. It is intentionally retained across episode
resets until the next scheduled refresh. Observation normalization state is
stored on the returned actor and is frozen by `predict`; evaluation uses the
environment's original rewards.

## TODO (Algorithms):
* ~~Simple DQN~~
* ~~A2C~~
* Rainbow
* ~~PPO~~
* ~~DDPG~~
* SAC

## TODO (Utils):
* ~~General replay buffer~~
* ~~CLI for hyperparameters~~
* ~~Support loggers~~
* GPU training
* ~~Multi-thread PPO~~
* Vectorized envs
* ~~Plotting~~
* ~~Multi-loggers (file/console/Tensorboard)~~
* ~~Make nn inputs F32 - F32 env wrapper?~~ - done for PPO
* Make individual file runners e.g experiments/run_ppo.(jl/sh)
* Better logging - log interval and always log at correct step

## TODO (Investigate):
* Profile PPO
