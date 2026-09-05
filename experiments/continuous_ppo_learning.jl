using CleanRL
using POMDPs
import Random

struct ContextDistribution end
struct ContextualQuadratic <: MDP{Float32,Vector{Float32}} end

Random.rand(rng::Random.AbstractRNG, ::ContextDistribution) =
  2f0 * rand(rng, Float32) - 1f0
POMDPs.initialstate(::ContextualQuadratic) = ContextDistribution()
POMDPs.isterminal(::ContextualQuadratic, ::Float32) = false
POMDPs.convert_s(::Type{Vector{Float32}}, s::Float32, ::ContextualQuadratic) = Float32[s]

target_action(s) = Float32[-0.5f0 + 0.75f0 * s, 1.5f0 - 0.75f0 * s]
POMDPs.gen(::ContextualQuadratic, s::Float32, action::Vector{Float32}, rng) =
  (sp=s, r=-sum(abs2, action .- target_action(s)))

const SPEC = BoxActionSpec(Float32[-2, 0], Float32[1, 3])
const EVALUATION_STATES = collect(range(-1f0, 1f0; length=201))

function evaluation_reward(actor)
  observations = reshape(EVALUATION_STATES, 1, :)
  actions = predict(actor, observations; deterministic=true)
  total = 0f0
  for i in eachindex(EVALUATION_STATES)
    total -= sum(abs2, view(actions, :, i) .- target_action(EVALUATION_STATES[i]))
  end
  total / length(EVALUATION_STATES)
end

function initial_actor(seed, use_sde)
  actor, _ = CleanRL.continuous_actor_critic(SPEC, 1;
    use_sde=use_sde, log_std_init=-2f0,
    rng=Random.MersenneTwister(seed))
  actor
end

function train(seed, use_sde, total_timesteps, log_dir)
  factory = (index, rng) -> MDPEnv(ContextualQuadratic();
    rng=rng, action_space=SPEC, max_steps=16)
  config = continuous_ppo_config(
    total_timesteps=total_timesteps,
    num_steps=32,
    num_envs=8,
    num_minibatches=4,
    update_epochs=4,
    normalize_observations=false,
    use_sde=use_sde,
    seed=seed,
    name="contextual-$(use_sde ? "gsde" : "gaussian")-seed-$seed",
    log_dir=log_dir,
  )
  first(ppo(factory, config))
end

total_timesteps = parse(Int, get(ENV, "CLEANRL_BENCH_STEPS", "20000"))
seeds = parse.(Int, split(get(ENV, "CLEANRL_BENCH_SEEDS", "1,2,3,4,5"), ','))
log_dir = get(ENV, "CLEANRL_BENCH_LOG_DIR", mktempdir())

println("mode,seed,initial_reward,trained_reward,gap_closed_percent")
results = Dict(false => Float32[], true => Float32[])
for use_sde in (false, true), seed in seeds
  initial_reward = evaluation_reward(initial_actor(seed, use_sde))
  actor = train(seed, use_sde, total_timesteps, log_dir)
  trained_reward = evaluation_reward(actor)
  gap_closed = 100f0 * (trained_reward - initial_reward) / -initial_reward
  mode = use_sde ? "gsde" : "gaussian"
  println("$mode,$seed,$initial_reward,$trained_reward,$gap_closed")
  push!(results[use_sde], trained_reward)
end

# A mean reward of -0.02 corresponds to a joint action RMSE below sqrt(0.02)
# over the fixed evaluation grid; the analytical optimum is zero.
const SUCCESS_THRESHOLD = -0.02f0
for use_sde in (false, true)
  values = results[use_sde]
  average = sum(values) / length(values)
  spread = length(values) == 1 ? 0f0 :
    sqrt(sum(abs2, values .- average) / (length(values) - 1))
  mode = use_sde ? "gsde" : "gaussian"
  println("summary,$mode,mean=$average,std=$spread,threshold=$SUCCESS_THRESHOLD")
  all(>=(SUCCESS_THRESHOLD), values) ||
    error("$mode failed the continuous-control learning threshold")
end
