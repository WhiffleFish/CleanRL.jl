using CleanRL
using POMDPs
import Random

"""A tiny native continuous-control example with a known target action."""
struct QuadraticControl <: MDP{Float32,Float32} end
struct InitialQuadraticState end

Random.rand(::Random.AbstractRNG, ::InitialQuadraticState) = 0f0
POMDPs.initialstate(::QuadraticControl) = InitialQuadraticState()
POMDPs.isterminal(::QuadraticControl, ::Float32) = false
POMDPs.convert_s(::Type{Vector{Float32}}, s::Float32, ::QuadraticControl) = Float32[s]
POMDPs.gen(::QuadraticControl, s::Float32, a::Float32, rng) =
  (sp=0.9f0 * s + 0.1f0 * a, r=-(a - 0.25f0)^2 - 0.01f0 * s^2)

spec = BoxActionSpec(-2f0, 2f0)
factory = (index, rng) -> MDPEnv(QuadraticControl();
  rng=rng, action_space=spec, action_decoder=only, max_steps=100)

# Change to PPOConfig(..., use_sde=false) for the matched Gaussian baseline.
config = continuous_ppo_config(total_timesteps=100_000, name="quadratic-gsde")
actor, critic = ppo(factory, config)
println("deterministic action at zero: ", only(predict(actor, Float32[0])))
