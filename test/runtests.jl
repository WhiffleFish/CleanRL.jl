using Test

using CleanRL
using POMDPModels
using POMDPs
import Random

make_mdp() = SimpleGridWorld()

function finishes(f)
  f()
  true
end


struct ContinuousInitialState end
struct ContinuousQuadraticMDP <: MDP{Float32,Float32} end
Random.rand(::Random.AbstractRNG, ::ContinuousInitialState) = 0f0
POMDPs.initialstate(::ContinuousQuadraticMDP) = ContinuousInitialState()
POMDPs.isterminal(::ContinuousQuadraticMDP, ::Float32) = false
POMDPs.actions(::ContinuousQuadraticMDP) = error("continuous actions must not be enumerated")
POMDPs.convert_s(::Type{Vector{Float32}}, s::Float32, ::ContinuousQuadraticMDP) = Float32[s]
POMDPs.gen(::ContinuousQuadraticMDP, s::Float32, a::Float32, rng) =
  (sp=s + a, r=-(a - 0.25f0)^2)

@testset "Box action adapter" begin
  @test_throws ArgumentError BoxActionSpec(Float32[], Float32[])
  @test_throws ArgumentError BoxActionSpec(Float32[0], Float32[0])
  @test_throws ArgumentError BoxActionSpec(Float32[-Inf], Float32[1])
  @test_throws ArgumentError BoxActionSpec(zeros(Float32, 2), ones(Float32, 3))

  spec = BoxActionSpec(-2f0, 3f0)
  env = MDPEnv(ContinuousQuadraticMDP(); action_space=spec,
    action_decoder=only, max_steps=1)
  @test action_space(env) == spec
  @test action_dim(env) == 1
  @test_throws ArgumentError action_count(env)
  @test -2f0 <= only(random_action(env)) <= 3f0
  step!(env, Float32[0.5])
  @test !is_true_terminated(env)
  @test is_truncated(env)
  @test is_terminated(env)

  shaped_spec = BoxActionSpec(zeros(Float32, 2, 2), ones(Float32, 2, 2))
  shaped_decoder = CleanRL._default_box_decoder(shaped_spec)
  @test shaped_spec isa BoxActionSpec{2}
  @test @inferred(shaped_decoder(Float32[1, 2, 3, 4])) == Float32[1 3; 2 4]
end

@testset "Continuous policies" begin
  spec = BoxActionSpec(Float32[-2, 0], Float32[2, 3])
  obs = reshape(Float32[0.2, -0.4, 1, 0, 0.5, -1], 3, 2)

  actor, _ = CleanRL.continuous_actor_critic(spec, 3; log_std_init=-0.7f0)
  @test fieldtype(typeof(actor), :action_shape) === Tuple{Int}
  actor_copy, _ = CleanRL.continuous_actor_critic(spec, 3; log_std_init=-0.7f0,
    rng=Random.MersenneTwister(1))
  actor_repeat, _ = CleanRL.continuous_actor_critic(spec, 3; log_std_init=-0.7f0,
    rng=Random.MersenneTwister(1))
  @test CleanRL.Flux.trainables(actor_copy) == CleanRL.Flux.trainables(actor_repeat)
  rng = Random.MersenneTwister(11)
  raw, old_logprob = sample_policy(rng, actor, obs, nothing)
  new_logprob, entropy = evaluate_policy(actor, obs, raw)
  @test size(raw) == (2, 2)
  @test old_logprob ≈ new_logprob
  @test all(isfinite, entropy)
  bounded = environment_action(actor, 10f0 .* raw)
  @test all(spec.low .<= bounded) && all(bounded .<= spec.high)
  @test length(CleanRL.Flux.trainables(actor)) == 7

  gsde, _ = CleanRL.continuous_actor_critic(spec, 3; use_sde=true,
    full_std=false, log_std_init=-1f0)
  exploration = ExplorationState(gsde, 2)
  reset_noise!(rng, exploration, gsde)
  gsde_raw, gsde_logprob = sample_policy(rng, gsde, obs, exploration)
  @test size(gsde_raw) == (2, 2)
  @test all(isfinite, gsde_logprob)
  @test_throws DimensionMismatch sample_policy(rng, gsde, obs[:, 1:1], exploration)

  tanh_actor, _ = CleanRL.continuous_actor_critic(spec, 3; action_transform=:tanh)
  saturated = Float32[100 -100; -100 100]
  base = Float32[-1, -1]
  @test all(isfinite, environment_logprob(tanh_actor, saturated; base_logprob=base))
  @test_throws ArgumentError environment_logprob(actor, raw; base_logprob=old_logprob)
end

@testset "Transition-indexed GAE" begin
  rewards = reshape(Float32[2, 2], 1, 2)
  values = reshape(Float32[1, 1], 1, 2)
  next_values = reshape(Float32[4, 4], 1, 2)
  terminal = reshape(Bool[true, false], 1, 2)
  timeout = reshape(Bool[false, true], 1, 2)
  advantages = gae_transitions(rewards, values, next_values, terminal, timeout, 0.5f0, 0.5f0)
  @test advantages == reshape(Float32[1, 3], 1, 2)
end

@testset "Continuous PPO smoke" begin
  spec = BoxActionSpec(-2f0, 2f0)
  factory = (index, rng) -> MDPEnv(ContinuousQuadraticMDP(); rng=rng,
    action_space=spec, action_decoder=only, max_steps=2)
  common = (total_timesteps=4, num_steps=2, num_envs=2,
    num_minibatches=1, update_epochs=1, ent_coeff=0f0,
    clip_value_loss=false, anneal_lr=false)
  gaussian, _ = ppo(factory, PPOConfig(; common..., name="test-continuous-gaussian"))
  gsde, _ = ppo(factory, PPOConfig(; common..., use_sde=true,
    log_std_init=-2f0, name="test-continuous-gsde"))
  @test gaussian isa DiagGaussianActor
  @test gsde isa GSDEActor
  @test -2f0 <= only(predict(gaussian, Float32[0])) <= 2f0
  @test -2f0 <= only(predict(gsde, Float32[0])) <= 2f0
end

@testset "MDPEnv smoke" begin
  mdp = make_mdp()
  env = MDPEnv(mdp; max_steps=1)

  @test state(env) isa Vector{Float32}
  @test state_dim(env) == 2
  @test action_count(env) == length(collect(POMDPs.actions(mdp)))
  @test random_action(env) in POMDPs.actions(mdp)
  @test !is_terminated(env)

  step!(env, 1)
  @test env.step_count == 1
  @test current_reward(env) isa Float32
  @test is_terminated(env)

  reset!(env)
  @test env.step_count == 0
  @test current_reward(env) == 0.0f0
  @test !is_terminated(env)
end

@testset "MultiThreadEnv smoke" begin
  env = MultiThreadEnv(() -> MDPEnv(make_mdp(); max_steps=1), 2)

  @test size(state(env)) == (2, 2)
  @test current_reward(env) == Float32[0, 0]
  @test is_terminated(env) == Bool[0, 0]

  step!(env, [1, 4])
  @test size(state(env)) == (2, 2)
  @test current_reward(env) isa Vector{Float32}
  @test is_terminated(env) == Bool[1, 1]

  reset!(env)
  @test size(state(env)) == (2, 2)
  @test is_terminated(env) == Bool[0, 0]
end

@testset "GAE" begin
  # γ=λ=0.5 and integer values/rewards keep every intermediate value an exact
  # binary fraction, so the expected advantages can be checked with `==`.
  γ = 0.5f0
  λ = 0.5f0

  @testset "no terminals" begin
    values = Float32[1, 1, 1, 1]      # [0, k], k=3
    rewards = Float32[2, 2, 2]        # [1, k]
    terminals = falses(4)             # [0, k]

    advantages = CleanRL.gae(values, rewards, terminals, γ, λ)

    @test length(advantages) == length(rewards)
    @test all(isfinite, advantages)
    # regression check for the bug where the loop skipped t == length(rewards),
    # leaving advantages[end] as whatever garbage `similar` handed back
    @test advantages[end] == 1.5f0
    @test advantages == Float32[1.96875, 1.875, 1.5]
  end

  @testset "terminal cuts bootstrap" begin
    values = Float32[1, 1, 1, 1]
    rewards = Float32[2, 2, 2]
    terminals = Bool[0, 0, 1, 0]       # episode ends after step 2

    advantages = CleanRL.gae(values, rewards, terminals, γ, λ)

    @test length(advantages) == length(rewards)
    @test all(isfinite, advantages)
    @test advantages[end] == 1.5f0
    @test advantages == Float32[1.75, 1.0, 1.5]
  end
end

@testset "Algorithm smoke" begin
  mdp = make_mdp()

  @test finishes(() -> dqn(mdp, DQNConfig(
    run_name="test-dqn",
    total_timesteps=3,
    min_buff_size=10,
  ); max_steps=1))

  @test finishes(() -> a2c(mdp, A2CConfig(
    run_name="test-a2c",
    total_timesteps=3,
    min_replay_size=10,
  ); max_steps=1))

  @test finishes(() -> ppo(mdp, PPOConfig(
    total_timesteps=8,
    num_steps=2,
    num_envs=2,
    num_minibatches=1,
    update_epochs=1,
    anneal_lr=true,
  ); max_steps=1))
end
