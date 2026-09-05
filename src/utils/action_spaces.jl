export AbstractActionSpec, DiscreteActionSpec, BoxActionSpec

abstract type AbstractActionSpec end

struct DiscreteActionSpec{A} <: AbstractActionSpec
  actions::Vector{A}
  function DiscreteActionSpec(actions)
    values = collect(actions)
    isempty(values) && throw(ArgumentError("a discrete action space must not be empty"))
    new{eltype(values)}(values)
  end
end

struct BoxActionSpec{N} <: AbstractActionSpec
  low::Vector{Float32}
  high::Vector{Float32}
  shape::NTuple{N,Int}
end

Base.:(==)(a::DiscreteActionSpec, b::DiscreteActionSpec) = a.actions == b.actions
Base.:(==)(a::BoxActionSpec, b::BoxActionSpec) =
  a.shape == b.shape && a.low == b.low && a.high == b.high

function BoxActionSpec(low, high)
  size(low) == size(high) || throw(ArgumentError("Box low and high must have matching shapes"))
  shape = low isa Number ? () : size(low)
  lo = _box_vector(low)
  hi = _box_vector(high)
  isempty(lo) && throw(ArgumentError("a Box action space must not be empty"))
  all(isfinite, lo) && all(isfinite, hi) ||
    throw(ArgumentError("Box bounds must be finite and representable as Float32"))
  all(hi .> lo) || throw(ArgumentError("every Box dimension must have positive width"))
  center = lo .+ (hi .- lo) ./ 2f0
  scale = (hi .- lo) ./ 2f0
  all(isfinite, center) && all(isfinite, scale) && all(scale .> 0f0) ||
    throw(ArgumentError("Box midpoint and scale must be finite and representable as Float32"))
  BoxActionSpec(lo, hi, shape)
end

function _box_vector(x)
  values = x isa Number ? [x] : vec(collect(x))
  converted = try
    Float32.(values)
  catch
    throw(ArgumentError("Box bounds must be representable as Float32"))
  end
  converted
end
