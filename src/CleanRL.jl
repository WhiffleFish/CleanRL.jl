module CleanRL

export DQNConfig, A2CConfig, DDPGConfig, PPOConfig, dqn, a2c, ddpg, ppo

using Base: Threads
using Dates: now, format
using Random: shuffle, Xoshiro

using POMDPs

using Flux
using Flux: Zygote
using StatsBase: sample, Weights, loglikelihood, mean, entropy, std
using Random: shuffle
using Distributions: Categorical, logpdf

using Dates: now, format

include("utils/replay_buffer.jl")
include("utils/config_parser.jl")
include("utils/logger.jl")
include("utils/pomdp_env.jl")
include("utils/networks.jl")
include("utils/multi_thread_env.jl")

include("algorithms/dqn.jl")
include("algorithms/ddpg.jl")
include("algorithms/a2c.jl")
include("algorithms/ppo.jl")

end # module
