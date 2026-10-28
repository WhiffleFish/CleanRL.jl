export continuous_ppo_config, gae_transitions

@kwdef struct PPOConfig
  total_timesteps::Int = 500_000
  num_steps::Int = 32
  num_envs::Int = 4
  num_minibatches::Int = 4
  update_epochs::Int = 4
  lr::Float32 = 2.5f-4
  gamma::Float32 = 0.99f0
  gae_lambda::Float32 = 0.95f0
  clip_coef::Float32 = 0.2f0
  ent_coeff::Float32 = 0.01f0
  v_coef::Float32 = 0.5f0
  max_grad_norm::Float32 = 0.5f0
  normalize_advantages::Bool = true
  normalize_observations::Bool = false
  normalize_rewards::Bool = false
  clip_value_loss::Bool = true
  anneal_lr::Bool = true
  use_sde::Bool = false
  sde_sample_freq::Int = -1
  log_std_init::Float32 = 0f0
  full_std::Bool = true
  action_transform::Symbol = :clip
  target_kl::Union{Nothing,Float32} = nothing
  seed::Int = 1
  name::String = "ppo-test"
  log_dir::String = "logs"
end

continuous_ppo_config(; kwargs...) = PPOConfig(; total_timesteps=500_000,
  num_steps=256, num_envs=8, num_minibatches=8, update_epochs=10,
  lr=3f-4, ent_coeff=0f0, clip_value_loss=false, use_sde=true,
  sde_sample_freq=4, log_std_init=-2f0, normalize_observations=true,
  target_kl=0.02f0, kwargs...)

function _validate_ppo(config::PPOConfig)
  config.total_timesteps > 0 || throw(ArgumentError("total_timesteps must be positive"))
  config.num_steps > 0 || throw(ArgumentError("num_steps must be positive"))
  config.num_envs > 0 || throw(ArgumentError("num_envs must be positive"))
  config.num_minibatches > 0 || throw(ArgumentError("num_minibatches must be positive"))
  config.update_epochs > 0 || throw(ArgumentError("update_epochs must be positive"))
  batch = config.num_steps * config.num_envs
  batch % config.num_minibatches == 0 || throw(ArgumentError("num_steps*num_envs must be divisible by num_minibatches"))
  minibatch = batch ÷ config.num_minibatches
  config.normalize_advantages && minibatch <= 1 &&
    throw(ArgumentError("advantage normalization requires minibatches larger than one"))
  config.total_timesteps >= batch || throw(ArgumentError("total_timesteps must contain at least one complete rollout ($batch steps)"))
  isfinite(config.lr) && config.lr > 0 || throw(ArgumentError("lr must be finite and positive"))
  0f0 <= config.gamma <= 1f0 || throw(ArgumentError("gamma must be in [0, 1]"))
  0f0 <= config.gae_lambda <= 1f0 || throw(ArgumentError("gae_lambda must be in [0, 1]"))
  isfinite(config.clip_coef) && config.clip_coef > 0 || throw(ArgumentError("clip_coef must be finite and positive"))
  isfinite(config.ent_coeff) && config.ent_coeff >= 0 || throw(ArgumentError("ent_coeff must be finite and nonnegative"))
  isfinite(config.v_coef) && config.v_coef >= 0 || throw(ArgumentError("v_coef must be finite and nonnegative"))
  isfinite(config.max_grad_norm) && config.max_grad_norm > 0 || throw(ArgumentError("max_grad_norm must be finite and positive"))
  isfinite(config.log_std_init) || throw(ArgumentError("log_std_init must be finite"))
  config.sde_sample_freq == -1 || config.sde_sample_freq > 0 ||
    throw(ArgumentError("sde_sample_freq must be -1 or a positive integer"))
  config.action_transform in (:clip, :tanh) || throw(ArgumentError("action_transform must be :clip or :tanh"))
  config.target_kl === nothing || (isfinite(config.target_kl) && config.target_kl > 0) ||
    throw(ArgumentError("target_kl must be nothing or finite and positive"))
  config
end

function get_action(obs::AbstractVecOrMat{Float32}, actor::Chain; rng=Random.default_rng())
  logits = actor(obs)
  probs = softmax(logits)
  logprobs = logsoftmax(logits)
  batch_size = last(size(obs))
  action = [sample(rng, Weights(view(probs, :, i))) for i in 1:batch_size]
  action, logprobs[CartesianIndex.(action, 1:batch_size)]
end

get_action(obs::AbstractVecOrMat{Float32}, actor::AbstractContinuousActor;
  rng=Random.default_rng(), exploration_state=nothing) = sample_policy(rng, actor, obs, exploration_state)

function logprob_actions(obs::AbstractVecOrMat{Float32}, actor::Chain,
  actions::AbstractVector{<:Integer})
  logits = actor(obs)
  probs = softmax(logits)
  logprobs = logsoftmax(logits)
  batch_size = last(size(obs))
  logprobs[CartesianIndex.(actions, 1:batch_size)], vec(-sum(probs .* logprobs; dims=1))
end

logprob_actions(obs::AbstractVecOrMat{Float32}, actor::AbstractContinuousActor,
  actions::AbstractVecOrMat) = evaluate_policy(actor, obs, actions)

function gae(values::AbstractVector{T}, rewards::AbstractVector{T},
  terminals::AbstractVector{Bool}, γ::T, λ::T) where {T<:AbstractFloat}
  advantages = similar(rewards)
  nonterm = 1.0f0 .- terminals
  lastgaelam = zero(T)
  for t in length(rewards):-1:1
    δ = rewards[t] + γ * nonterm[t+1] * values[t+1] - values[t]
    lastgaelam = δ + γ * λ * nonterm[t+1] * lastgaelam
    advantages[t] = lastgaelam
  end
  advantages
end

"""Transition-indexed GAE with distinct terminal and time-limit masks."""
function gae_transitions(rewards::AbstractMatrix{T}, values::AbstractMatrix{T},
  next_values::AbstractMatrix{T}, terminated::AbstractMatrix{Bool},
  truncated::AbstractMatrix{Bool}, γ::T, λ::T) where {T<:AbstractFloat}
  size(rewards) == size(values) == size(next_values) == size(terminated) == size(truncated) ||
    throw(DimensionMismatch("all transition GAE arrays must have the same shape"))
  advantages = similar(rewards)
  next_advantage = zeros(T, size(rewards, 1))
  for t in reverse(axes(rewards, 2))
    bootstrap_mask = .!view(terminated, :, t)
    continuation_mask = .!(view(terminated, :, t) .| view(truncated, :, t))
    δ = view(rewards, :, t) .+ γ .* bootstrap_mask .* view(next_values, :, t) .- view(values, :, t)
    advantages[:, t] .= δ .+ γ .* λ .* continuation_mask .* next_advantage
    next_advantage .= view(advantages, :, t)
  end
  advantages
end

ppo(mdp::POMDPs.MDP, config::PPOConfig=PPOConfig(); kwargs...) =
  ppo(() -> MDPEnv(mdp; kwargs...), config)

_normalize_training_obs(rms, raw) = rms === nothing ? Float32.(raw) : normalize(rms, raw)

function _finite_or_error(name, x)
  all(isfinite, x) || throw(ErrorException("nonfinite $name encountered; update rejected"))
  x
end

_finite_gradient(x::Nothing) = true
_finite_gradient(x::Number) = isfinite(x)
_finite_gradient(x::AbstractArray) = all(isfinite, x)
function _finite_gradient(x)
  children = Flux.Functors.children(x)
  all(_finite_gradient, values(children))
end

function _make_ppo_env(env_factory, index::Int, seed::Int)
  env_rng = Random.MersenneTwister(seed + index)
  if applicable(env_factory, index, env_rng)
    env_factory(index, env_rng)
  elseif applicable(env_factory, env_rng)
    env_factory(env_rng)
  else
    env_factory()
  end
end

"""
    ppo(env_factory, config)

Build `config.num_envs` environments with `env_factory` and train on them. The factory is
called as `f(index, rng)`, `f(rng)` or `f()`, whichever it accepts, with a per-environment
`MersenneTwister(config.seed + index)`. Keyword arguments are forwarded to `ppo(env, config)`.
"""
ppo(env_factory::Function, config::PPOConfig=PPOConfig(); kwargs...) =
  (_validate_ppo(config);
   ppo(MultiThreadEnv([_make_ppo_env(env_factory, i, config.seed) for i in 1:config.num_envs]),
       config; kwargs...))

"""
    ppo(env, config)

Train on an already-constructed vectorized environment.

`env` needs `length`, `single_state_dim`, `single_action_space`, `single_action_dim` (Box
spaces only), `state`, `step!`, `reset!(env; is_force)`, `current_reward`,
`is_true_terminated` and `is_truncated` -- it does NOT have to be a `MultiThreadEnv` or hold
`MDPEnv`s. That is the entry point for an environment whose vectorization is the point: one
that evaluates a network over all `num_envs` states in a single batched forward pass, rather
than once per environment.

`config.seed` still seeds the actor initialization and the update loop, but a prebuilt env
carries its own RNG and this method does not reseed it -- seeding it is the caller's job.

`initial_actor` and `initial_critic` warm-start training from copies of existing networks
(for example, a behavior-cloned policy) instead of a fresh initialization. They must accept
the raw observations, so they cannot be combined with `normalize_observations`.
"""
function ppo(env, config::PPOConfig; initial_actor=nothing, initial_critic=nothing)
  _validate_ppo(config)
  length(env) == config.num_envs ||
    throw(ArgumentError("env has $(length(env)) environments but config.num_envs = $(config.num_envs)"))
  warm_start = initial_actor !== nothing || initial_critic !== nothing
  warm_start && (initial_actor === nothing || initial_critic === nothing) &&
    throw(ArgumentError("warm-starting PPO requires both initial_actor and initial_critic"))
  warm_start && config.normalize_observations &&
    throw(ArgumentError("initial networks cannot be combined with normalize_observations"))
  rng = Random.MersenneTwister(config.seed)
  init_rng = Random.MersenneTwister(config.seed)
  spec = single_action_space(env)
  config.use_sde && spec isa DiscreteActionSpec && throw(ArgumentError("use_sde=true requires a BoxActionSpec"))
  spec isa BoxActionSpec && config.action_transform === :tanh && config.ent_coeff != 0f0 &&
    throw(ArgumentError("tanh continuous policies currently require ent_coeff=0"))
  Logger.make_logger(config.name; log_dir=config.log_dir, to_terminal=false)

  obs_dim = single_state_dim(env)
  obs_rms = config.normalize_observations ? RunningMeanStd(obs_dim) : nothing
  actor, critic = if warm_start
    deepcopy(initial_actor), deepcopy(initial_critic)
  elseif spec isa BoxActionSpec
    continuous_actor_critic(spec, obs_dim; use_sde=config.use_sde,
      log_std_init=config.log_std_init, full_std=config.full_std,
      action_transform=config.action_transform, obs_normalizer=obs_rms,
      rng=init_rng)
  else
    Networks.make_actor_critic(single_action_count(env), obs_dim; rng=init_rng)
  end
  actor, critic = Flux.f32(actor), Flux.f32(critic)

  batch_size = config.num_steps * config.num_envs
  minibatch_size = batch_size ÷ config.num_minibatches
  num_updates = config.total_timesteps ÷ batch_size
  opt = Flux.OptimiserChain(ClipNorm(config.max_grad_norm),
    Adam(config.lr, (0.9f0, 0.999f0), 1f-5))
  opt_state = Flux.setup(opt, (actor, critic))
  n, tmax = config.num_envs, config.num_steps
  observations = zeros(Float32, obs_dim, n, tmax)
  actions = spec isa BoxActionSpec ? zeros(Float32, single_action_dim(env), n, tmax) : zeros(Int, n, tmax)
  logprobs = zeros(Float32, n, tmax)
  rewards = zeros(Float32, n, tmax)
  values = zeros(Float32, n, tmax)
  next_values = zeros(Float32, n, tmax)
  terminated = falses(n, tmax)
  truncated = falses(n, tmax)
  reward_normalizer = config.normalize_rewards ? RewardNormalizer(n, config.gamma) : nothing
  exploration = actor isa GSDEActor ? ExplorationState(actor, n) : nothing

  global_step = 0
  last_log_step = 0
  episode_returns = zeros(Float32, n)
  episode_lengths = zeros(Int, n)
  start_time = time()
  reset!(env; is_force=true)
  raw_next_obs = copy(state(env))

  for update in 1:num_updates
    if config.anneal_lr
      frac = 1f0 - Float32(update - 1) / Float32(num_updates)
      Flux.Optimisers.adjust!(opt_state, frac * config.lr)
    end
    exploration === nothing || reset_noise!(rng, exploration, actor)
    clipped_coordinates = 0
    clipped_transitions = 0
    for t in 1:tmax
      global_step += n
      episode_lengths .+= 1
      obs_rms === nothing || update!(obs_rms, raw_next_obs)
      obs = _normalize_training_obs(obs_rms, raw_next_obs)
      observations[:, :, t] .= obs
      if exploration !== nothing && config.sde_sample_freq > 0 && t > 1 && (t - 1) % config.sde_sample_freq == 0
        reset_noise!(rng, exploration, actor)
      end
      sampled_action, log_prob = actor isa Chain ? get_action(obs, actor; rng=rng) :
        get_action(obs, actor; rng=rng, exploration_state=exploration)
      values[:, t] .= vec(critic(obs))
      logprobs[:, t] .= log_prob
      executed_action = if actor isa AbstractContinuousActor
        actions[:, :, t] .= sampled_action
        clipped_coordinates += count(abs.(sampled_action) .> 1f0)
        clipped_transitions += count(vec(any(abs.(sampled_action) .> 1f0; dims=1)))
        environment_action(actor, sampled_action)
      else
        actions[:, t] .= sampled_action
        sampled_action
      end
      step!(env, executed_action)
      raw_rewards = copy(current_reward(env))
      final_obs = copy(state(env))
      term = copy(is_true_terminated(env))
      trunc_mask = copy(is_truncated(env))
      done = term .| trunc_mask
      terminated[:, t] .= term
      truncated[:, t] .= trunc_mask
      next_values[:, t] .= vec(critic(_normalize_training_obs(obs_rms, final_obs)))
      rewards[:, t] .= reward_normalizer === nothing ? raw_rewards :
        normalize_rewards!(reward_normalizer, raw_rewards, done)
      episode_returns .+= raw_rewards
      if any(done)
        steps_per_sec = trunc(Int, global_step / max(time() - start_time, eps()))
        for i in eachindex(done)
          done[i] || continue
          episode_return = episode_returns[i]
          episode_length = episode_lengths[i]
          log_step_inc = last_log_step == 0 ? 0 : global_step - last_log_step
          @info "Episode Statistics" episode_return episode_length global_step steps_per_sec log_step_increment=log_step_inc
          episode_returns[i] = 0f0
          episode_lengths[i] = 0
          last_log_step = global_step
        end
        reset!(env)
        raw_next_obs = copy(state(env))
      else
        raw_next_obs = final_obs
      end
    end

    advantages = gae_transitions(rewards, values, next_values, terminated, truncated,
      config.gamma, config.gae_lambda)
    returns = advantages .+ values
    states_flat = reshape(observations, obs_dim, batch_size)
    actions_flat = spec isa BoxActionSpec ? reshape(actions, single_action_dim(env), batch_size) : vec(actions)
    logprobs_flat, values_flat = vec(logprobs), vec(values)
    advantages_flat, returns_flat = vec(advantages), vec(returns)
    b_inds = collect(1:batch_size)
    stop_for_kl = false
    for _ in 1:config.update_epochs
      Random.shuffle!(rng, b_inds)
      for start in 1:minibatch_size:batch_size
        mb_inds = view(b_inds, start:(start + minibatch_size - 1))
        pg_loss = 0f0; v_loss = 0f0; entropy_loss = 0f0; approx_kl = 0f0
        loss, gs = Flux.withgradient((actor, critic)) do (actor_model, critic_model)
          mb_states = view(states_flat, :, mb_inds)
          mb_actions = spec isa BoxActionSpec ? view(actions_flat, :, mb_inds) : view(actions_flat, mb_inds)
          mb_adv = view(advantages_flat, mb_inds)
          mb_oldlogp = view(logprobs_flat, mb_inds)
          mb_oldvalues = view(values_flat, mb_inds)
          mb_returns = view(returns_flat, mb_inds)
          newlogprob, entropy = logprob_actions(mb_states, actor_model, mb_actions)
          newvalue = vec(critic_model(mb_states))
          mb_adv_norm = config.normalize_advantages ?
            (mb_adv .- mean(mb_adv)) ./ (std(mb_adv) .+ 1f-8) : mb_adv
          logratio = newlogprob .- mb_oldlogp
          ratio = _finite_or_error("likelihood ratio", exp.(logratio))
          approx_kl = mean(expm1.(logratio) .- logratio)
          pg_loss = mean(max.(-mb_adv_norm .* ratio,
            -mb_adv_norm .* clamp.(ratio, 1f0 - config.clip_coef, 1f0 + config.clip_coef)))
          v_loss = if config.clip_value_loss
            unclipped = (newvalue .- mb_returns) .^ 2
            clipped_value = mb_oldvalues .+ clamp.(newvalue .- mb_oldvalues, -config.clip_coef, config.clip_coef)
            0.5f0 * mean(max.(unclipped, (clipped_value .- mb_returns) .^ 2))
          else
            0.5f0 * mean((newvalue .- mb_returns) .^ 2)
          end
          entropy_loss = mean(entropy)
          total = pg_loss - config.ent_coeff * entropy_loss + config.v_coef * v_loss
          isfinite(total) || throw(ErrorException("nonfinite PPO loss encountered; update rejected"))
          total
        end
        if config.target_kl !== nothing && approx_kl > 1.5f0 * config.target_kl
          stop_for_kl = true
          break
        end
        _finite_gradient(gs[1]) || throw(ErrorException("nonfinite PPO gradient encountered; update rejected"))
        Flux.update!(opt_state, (actor, critic), gs[1])
        log_step_inc = last_log_step == 0 ? 0 : global_step - last_log_step
        @info "Training Statistics" loss pg_loss v_loss entropy_loss approx_kl log_step_increment=log_step_inc
        last_log_step = global_step
      end
      stop_for_kl && break
    end
    if actor isa AbstractContinuousActor
      @info "Continuous Policy Statistics" clipping_fraction=(clipped_coordinates / length(actions)) clipped_transition_fraction=(clipped_transitions / batch_size) epsilon_var=(actor isa GSDEActor ? actor.epsilon_var : 0f0)
    end
  end
  actor, critic
end

ppo(config::PPOConfig=PPOConfig()) =
  throw(ArgumentError("ppo requires a POMDPs.MDP or environment factory"))
