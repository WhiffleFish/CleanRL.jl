export MDPEnv, state, step!, reset!, is_terminated, is_true_terminated,
  is_truncated, current_reward, action_count, action_dim, state_dim,
  random_action, action_space, single_action_space, single_state_dim,
  single_action_count

mutable struct MDPEnv{M,RNG,S,AS,SE,AD,RS}
  mdp::M
  rng::RNG
  state::S
  reward::Float32
  terminal::Bool
  truncated::Bool
  step_count::Int
  max_steps::Union{Nothing,Int}
  action_spec::AS
  state_encoder::SE
  action_decoder::AD
  random_action_sampler::RS
end

_as_float_vector(x::AbstractArray) = Float32.(vec(x))
_as_float_vector(x::Number) = Float32[x]

struct _Omitted end
const _OMITTED = _Omitted()

function _default_box_decoder(spec::BoxActionSpec)
  spec.shape == () && return only
  length(spec.shape) == 1 && return a -> collect(a)
  a -> reshape(collect(a), spec.shape)
end

function MDPEnv(mdp::POMDPs.MDP; rng=Random.default_rng(),
  max_steps::Union{Nothing,Int}=nothing, action_space=nothing,
  actions=_OMITTED, state_encoder=s -> POMDPs.convert_s(Vector{Float32}, s, mdp),
  action_decoder=_OMITTED, random_action_sampler=_OMITTED)
  max_steps === nothing || max_steps > 0 || throw(ArgumentError("max_steps must be positive"))

  spec = if action_space === nothing
    DiscreteActionSpec(actions isa _Omitted ? POMDPs.actions(mdp) : actions)
  elseif action_space isa AbstractActionSpec
    actions isa _Omitted || throw(ArgumentError("pass either action_space or actions, not both"))
    action_space
  else
    throw(ArgumentError("action_space must be a DiscreteActionSpec or BoxActionSpec"))
  end

  decoder = if action_decoder isa _Omitted
    spec isa DiscreteActionSpec ? (a -> spec.actions[Int(a)]) : _default_box_decoder(spec)
  else
    action_decoder
  end
  sampler = if random_action_sampler isa _Omitted
    spec isa DiscreteActionSpec ? (r -> rand(r, spec.actions)) :
      (r -> spec.low .+ rand(r, Float32, length(spec.low)) .* (spec.high .- spec.low))
  else
    random_action_sampler
  end

  s = rand(rng, POMDPs.initialstate(mdp))
  terminal = POMDPs.isterminal(mdp, s)
  MDPEnv(mdp, rng, s, 0.0f0, terminal, false, 0, max_steps, spec,
    state_encoder, decoder, sampler)
end

state(env::MDPEnv) = env.state_encoder(env.state)
current_reward(env::MDPEnv) = env.reward
is_true_terminated(env::MDPEnv) = env.terminal
is_truncated(env::MDPEnv) = env.truncated
is_terminated(env::MDPEnv) = env.terminal || env.truncated
action_space(env::MDPEnv) = env.action_spec
action_count(env::MDPEnv{M,RNG,S,<:DiscreteActionSpec}) where {M,RNG,S} = length(env.action_spec.actions)
action_count(env::MDPEnv{M,RNG,S,<:BoxActionSpec}) where {M,RNG,S} =
  throw(ArgumentError("action_count is only defined for discrete action spaces; use action_dim"))
state_dim(env::MDPEnv) = length(state(env))
action_dim(env::MDPEnv{M,RNG,S,<:BoxActionSpec}) where {M,RNG,S} = length(env.action_spec.low)
action_dim(env::MDPEnv) = length(_as_float_vector(random_action(env)))
random_action(env::MDPEnv) = env.random_action_sampler(env.rng)

function reset!(env::MDPEnv)
  env.state = rand(env.rng, POMDPs.initialstate(env.mdp))
  env.reward = 0.0f0
  env.terminal = POMDPs.isterminal(env.mdp, env.state)
  env.truncated = false
  env.step_count = 0
  env
end

function step!(env::MDPEnv, action)
  is_terminated(env) && return env
  decoded_action = env.action_decoder(action)
  sp, r = POMDPs.@gen(:sp, :r)(env.mdp, env.state, decoded_action, env.rng)
  env.state = sp
  env.reward = Float32(r)
  env.step_count += 1
  env.terminal = POMDPs.isterminal(env.mdp, sp)
  env.truncated = !env.terminal && env.max_steps !== nothing && env.step_count >= env.max_steps
  env
end

function single_state_dim(env)
  state_dim(first(env.envs))
end

function single_action_count(env)
  action_count(first(env.envs))
end

single_action_space(env) = action_space(first(env.envs))
