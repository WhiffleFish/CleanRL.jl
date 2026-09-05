module Networks

using Flux
using Random

function mlp(layer_sizes::Vector{Int}, activation=tanh_fast, gain=sqrt(2);
  rng=Random.default_rng())
  layers = []
  for i in 1:length(layer_sizes)-1
    init = Flux.orthogonal(rng; gain=gain)
    push!(layers, Dense(layer_sizes[i], layer_sizes[i+1], activation; init=init))
  end
  Chain(layers...)
end

function make_actor_critic_shared(action_count::Int, obs_dim::Int,
  hidden_sizes::Vector{Int}=Int[64, 64]; rng=Random.default_rng())
  in_size = obs_dim
  out_size = action_count
  ob_net = mlp(vcat(in_size, hidden_sizes); rng=rng)

  actor_final_gain = Flux.orthogonal(rng; gain=0.01)
  critic_final_gain = Flux.orthogonal(rng; gain=1.0)
  actor_final_layer = Dense(last(hidden_sizes), out_size; init=actor_final_gain)
  critic_final_layer = Dense(last(hidden_sizes), 1; init=critic_final_gain)

  actor = Chain(ob_net, actor_final_layer)
  critic = Chain(ob_net, critic_final_layer)

  actor, critic
end

function make_actor_critic(action_count::Int, obs_dim::Int,
  hidden_sizes::Vector{Int}=Int[64, 64]; rng=Random.default_rng())
  in_size = obs_dim
  out_size = action_count
  actor_final_gain = Flux.orthogonal(rng; gain=0.01)
  critic_final_gain = Flux.orthogonal(rng; gain=1.0)
  actor_final_layer = Dense(last(hidden_sizes), out_size; init=actor_final_gain)
  critic_final_layer = Dense(last(hidden_sizes), 1; init=critic_final_gain)

  actor = Chain(mlp(vcat(in_size, hidden_sizes); rng=rng), actor_final_layer)
  critic = Chain(mlp(vcat(in_size, hidden_sizes); rng=rng), critic_final_layer)

  actor, critic
end

end  # module
