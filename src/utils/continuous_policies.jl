export AbstractContinuousActor, DiagGaussianActor, GSDEActor, ExplorationState,
  sample_policy, evaluate_policy, reset_noise!, predict, environment_action,
  environment_logprob, continuous_actor_critic

abstract type AbstractContinuousActor end

struct DiagGaussianActor{T,M,O,N} <: AbstractContinuousActor
  trunk::T
  mean_head::M
  log_std::Matrix{Float32}
  low::Vector{Float32}
  high::Vector{Float32}
  center::Vector{Float32}
  scale::Vector{Float32}
  action_shape::NTuple{N,Int}
  transform::Symbol
  obs_normalizer::O
end

struct GSDEActor{T,M,O,N} <: AbstractContinuousActor
  trunk::T
  mean_head::M
  log_std::Matrix{Float32}
  low::Vector{Float32}
  high::Vector{Float32}
  center::Vector{Float32}
  scale::Vector{Float32}
  action_shape::NTuple{N,Int}
  transform::Symbol
  epsilon_var::Float32
  obs_normalizer::O
end

Flux.@layer DiagGaussianActor trainable=(trunk, mean_head, log_std)
Flux.@layer GSDEActor trainable=(trunk, mean_head, log_std)

mutable struct ExplorationState
  weights::Array{Float32,3} # feature × action × environment
end

ExplorationState(actor::GSDEActor, batch_size::Integer) =
  ExplorationState(zeros(Float32, size(actor.log_std, 1), size(actor.mean_head.weight, 1), batch_size))

function _bounds(spec::BoxActionSpec)
  center = spec.low .+ (spec.high .- spec.low) ./ 2f0
  scale = (spec.high .- spec.low) ./ 2f0
  spec.low, spec.high, center, scale
end

function continuous_actor_critic(spec::BoxActionSpec, obs_dim::Int;
  use_sde=false, log_std_init=0f0, full_std=true, action_transform=:clip,
  obs_normalizer=nothing, hidden_sizes=Int[64, 64], rng=Random.default_rng())
  action_transform in (:clip, :tanh) || throw(ArgumentError("action_transform must be :clip or :tanh"))
  feature_dim = last(hidden_sizes)
  trunk = Networks.mlp(vcat(obs_dim, hidden_sizes); rng=rng)
  mean_head = Dense(feature_dim, length(spec.low); init=Flux.orthogonal(rng; gain=0.01))
  low, high, center, scale = _bounds(spec)
  actor = if use_sde
    scale_cols = full_std ? length(spec.low) : 1
    GSDEActor(trunk, mean_head, fill(Float32(log_std_init), feature_dim, scale_cols),
      low, high, center, scale, spec.shape, action_transform, 1f-6, obs_normalizer)
  else
    DiagGaussianActor(trunk, mean_head, fill(Float32(log_std_init), length(spec.low), 1),
      low, high, center, scale, spec.shape, action_transform, obs_normalizer)
  end
  critic = Chain(Networks.mlp(vcat(obs_dim, hidden_sizes); rng=rng),
    Dense(feature_dim, 1; init=Flux.orthogonal(rng; gain=1.0)))
  actor, critic
end

_features_and_mean(actor::AbstractContinuousActor, obs) = begin
  h = actor.trunk(obs)
  h, actor.mean_head(h)
end

function _gaussian_stats(actor::DiagGaussianActor, obs)
  _, μ = _features_and_mean(actor, obs)
  variance = exp.(2f0 .* actor.log_std) .+ zero(eltype(μ)) .* μ
  μ, variance
end

function _gaussian_stats(actor::GSDEActor, obs)
  h, μ = _features_and_mean(actor, obs)
  detached_h = Flux.ignore_derivatives() do
    h
  end
  scales2 = exp.(2f0 .* actor.log_std)
  if size(scales2, 2) == 1
    scales2 = repeat(scales2, 1, size(μ, 1))
  end
  variance = transpose(scales2) * (detached_h .^ 2) .+ actor.epsilon_var
  μ, variance
end

function _logprob_entropy(μ, variance, raw_actions)
  c_log2pi = Float32(log(2f0 * Float32(pi)))
  c_log2pie = Float32(log(2f0 * Float32(pi) * Float32(MathConstants.e)))
  logprob = vec(-0.5f0 .* sum(c_log2pi .+ log.(variance) .+
    (raw_actions .- μ) .^ 2 ./ variance; dims=1))
  entropy = vec(0.5f0 .* sum(c_log2pie .+ log.(variance); dims=1))
  logprob, entropy
end

function evaluate_policy(actor::AbstractContinuousActor, obs::AbstractVecOrMat, raw_actions::AbstractVecOrMat)
  μ, variance = _gaussian_stats(actor, obs)
  actions = ndims(raw_actions) == 1 ? reshape(raw_actions, :, 1) : raw_actions
  _logprob_entropy(μ, variance, actions)
end

function sample_policy(rng::Random.AbstractRNG, actor::DiagGaussianActor, obs::AbstractVecOrMat, exploration_state=nothing)
  μ, variance = _gaussian_stats(actor, obs)
  raw = μ .+ sqrt.(variance) .* randn(rng, Float32, size(μ))
  raw, first(evaluate_policy(actor, obs, raw))
end

function reset_noise!(rng::Random.AbstractRNG, state::ExplorationState, actor::GSDEActor; env_indices=:all)
  inds = env_indices === :all ? axes(state.weights, 3) : env_indices
  size(state.weights, 1) == size(actor.log_std, 1) || throw(DimensionMismatch("exploration feature size mismatch"))
  scales = exp.(actor.log_std)
  for e in inds
    z = randn(rng, Float32, size(state.weights, 1), size(state.weights, 2))
    state.weights[:, :, e] .= z .* scales
  end
  state
end

function sample_policy(rng::Random.AbstractRNG, actor::GSDEActor, obs::AbstractVecOrMat, state::ExplorationState)
  h, μ = _features_and_mean(actor, obs)
  batch = size(μ, 2)
  size(state.weights, 3) == batch || throw(DimensionMismatch("gSDE exploration batch size changed; create a matching ExplorationState"))
  structured = similar(μ)
  for e in 1:batch
    structured[:, e] = transpose(state.weights[:, :, e]) * h[:, e]
  end
  raw = μ .+ structured .+ sqrt(actor.epsilon_var) .* randn(rng, Float32, size(μ))
  raw, first(evaluate_policy(actor, obs, raw))
end

function environment_action(actor::AbstractContinuousActor, raw::AbstractVecOrMat)
  bounded = actor.transform === :clip ? clamp.(raw, -1f0, 1f0) : tanh.(raw)
  actor.center .+ actor.scale .* bounded
end

_stable_softplus(x) = max(x, zero(x)) + log1p(exp(-abs(x)))

"""Log density in environment units for the invertible tanh transform.

PPO itself uses latent-action log probabilities, whose fixed transform Jacobian
cancels from the old/new ratio. Clipped policies have boundary mass and therefore
do not expose an ordinary environment-action density.
"""
function environment_logprob(actor::AbstractContinuousActor, raw::AbstractVecOrMat;
  base_logprob=nothing, obs=nothing)
  actor.transform === :tanh || throw(ArgumentError("clipped policies do not have an ordinary environment-action density"))
  base = base_logprob === nothing ? first(evaluate_policy(actor, obs, raw)) : base_logprob
  u = ndims(raw) == 1 ? reshape(raw, :, 1) : raw
  log_sech2 = @. 2f0 * (log(2f0) - u - _stable_softplus(-2f0 * u))
  vec(base) .- vec(sum(log.(actor.scale) .+ log_sech2; dims=1))
end


function _prediction_observation(actor::AbstractContinuousActor, obs)
  x = Float32.(obs)
  actor.obs_normalizer === nothing ? x : normalize(actor.obs_normalizer, x)
end

function predict(actor::AbstractContinuousActor, obs; deterministic=true,
  exploration_state=nothing, rng=Random.default_rng())
  x = _prediction_observation(actor, obs)
  was_vector = ndims(x) == 1
  batch = was_vector ? reshape(x, :, 1) : x
  raw = if deterministic
    last(_features_and_mean(actor, batch))
  else
    first(sample_policy(rng, actor, batch, exploration_state))
  end
  result = environment_action(actor, raw)
  was_vector ? vec(result) : result
end

(actor::AbstractContinuousActor)(obs) = predict(actor, obs; deterministic=true)
