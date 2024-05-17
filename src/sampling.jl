#using LinearAlgebra, FastGaussQuadrature
"""
        function VibDistribution(x̄::Vector{T}, σₓ::Vector{T}, p̄::Vector{T}, σₚ::Vector{T}) where T  

Return the normal distribution for a system with `D` vibrational degrees of freedom.  

`x̄`: Expectation values for x; \\
`σₓ`: Standard variance for x; \\
`p̄`: Expectation values for p; \\
`σₚ`: Standard variance for p;
`D`: Number of classical vibrational degrees of freedom; \\
"""
struct VibDistribution{S<:Sampleable{Univariate, Continuous}}
    x::Vector{S}
    p::Vector{S}
    function VibDistribution(x̄::Vector{T}, σₓ::Vector{T}, p̄::Vector{T}, σₚ::Vector{T}) where T
        D = length(x̄)
        D == length(σₓ) == length(p̄) == length(σₚ) || throw(DimensionMismatch("the length of the expectation values x̄, p̄, and standard variances σₓ, σₚ should be consistent"))
        x = [Normal{T}(x̄[i], σₓ[i]) for i=1:D]
        p = [Normal{T}(p̄[i], σₚ[i]) for i=1:D]
        new{Normal{T}}(x,p)
    end
    function VibDistribution(x̄::T, σₓ::T, p̄::T, σₚ::T, D::Int) where T
        x = [Normal{T}(x̄, σₓ) for i=1:D]
        p = [Normal{T}(p̄, σₚ) for i=1:D]
        new{Normal{T}}(x,p)
    end
end

function Random.rand(rng::AbstractRNG, d::SamplerTrivial{<:VibDistribution})
    x = similar(d[].x, eltype(eltype(d[].x)))
    for I in eachindex(x, d[].x)
        x[I] = rand(rng, d[].x[I])
    end
    p = similar(d[].p, eltype(eltype(d[].p)))
    for I in eachindex(p, d[].p)
        p[I] = rand(rng, d[].p[I])
    end
    return Vibs(x,p)
end

mutable struct Vibs{T}
    x::Vector{T}
    p::Vector{T}
    function Vibs(N::Int, T::Type=Float64)
        x = zeros(T, N)
        p = zeros(T, N)
        new{T}(x, p)
    end
    function Vibs(x::Vector{T}, p::Vector{T}) where {T}
        new{T}(x, p)
    end
end


function Base.similar(vib::Vibs)
    N = length(vib.x)
    x = zeros(eltype(vib.x), N)
    p = zeros(eltype(vib.p), N)
    return Vibs(x, p)
end

function Base.:+(vib1::Vibs, vib2::Vibs)
    length(vib1.x) == length(vib2.x) || throw(DimensionMismatch("The length of two Vibs structs should be the same!"))
    x = vib1.x .+ vib2.x
    p = vib1.p .+ vib2.p
    return Vibs(x, p)
end

function Base.:-(vib1::Vibs, vib2::Vibs)
    length(vib1.x) == length(vib2.x) || throw(DimensionMismatch("The length of two Vibs structs should be the same!"))
    x = vib1.x .- vib2.x
    p = vib1.p .- vib2.p
    return Vibs(x, p)
end

function Base.:*(vib::Vibs, c::Number)
    x = vib.x .* c 
    p = vib.p .* c
    return Vibs(x, p)
end

function Base.:*(c::Number, vib::Vibs)
    x = vib.x .* c 
    p = vib.p .* c
    return Vibs(x, p)
end
