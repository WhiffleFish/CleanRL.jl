export RunningMeanStd, update!, normalize, RewardNormalizer, normalize_rewards!

mutable struct RunningMeanStd
  mean::Vector{Float64}
  var::Vector{Float64}
  count::Float64
end


RunningMeanStd(dim::Integer) = RunningMeanStd(zeros(dim), ones(dim), 1e-4)

function update!(rms::RunningMeanStd, x::AbstractVecOrMat)
  data = ndims(x) == 1 ? reshape(x, :, 1) : x
  size(data, 1) == length(rms.mean) || throw(DimensionMismatch("running-statistic dimension mismatch"))
  n = size(data, 2)
  n == 0 && return rms
  batch_mean = vec(sum(Float64.(data); dims=2)) ./ n
  centered = Float64.(data) .- batch_mean
  batch_var = vec(sum(centered .^ 2; dims=2)) ./ n
  delta = batch_mean .- rms.mean
  total = rms.count + n
  m2 = rms.var .* rms.count .+ batch_var .* n .+ delta .^ 2 .* (rms.count * n / total)
  rms.mean .= rms.mean .+ delta .* (n / total)
  rms.var .= m2 ./ total
  rms.count = total
  rms
end

function normalize(rms::RunningMeanStd, x::AbstractArray; clip::Real=10f0)
  μ = Float32.(rms.mean)
  σ = Float32.(sqrt.(rms.var .+ 1e-8))
  result = ndims(x) == 1 ? (Float32.(x) .- μ) ./ σ :
    (Float32.(x) .- μ) ./ σ
  clamp.(result, -Float32(clip), Float32(clip))
end

mutable struct RewardNormalizer
  returns::Vector{Float64}
  rms::RunningMeanStd
  gamma::Float32
end

RewardNormalizer(n::Integer, gamma::Real) = RewardNormalizer(zeros(n), RunningMeanStd(1), Float32(gamma))

function normalize_rewards!(normalizer::RewardNormalizer, rewards, done)
  normalizer.returns .= normalizer.gamma .* normalizer.returns .+ Float64.(rewards)
  update!(normalizer.rms, reshape(normalizer.returns, 1, :))
  result = Float32.(rewards) ./ Float32(sqrt(normalizer.rms.var[1] + 1e-8))
  normalizer.returns[done] .= 0.0
  result
end

