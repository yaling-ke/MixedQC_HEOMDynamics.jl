s⁺ = [0. 0.; 1. 0.]
s⁻ = Matrix(s⁺')
sup = [1. 0.; 0. 0.]
sdn = [0. 0.; 0. 1.]
sz = [1. 0.; 0. -1.]
sx = [0 1; 1 0]
a⁺(N) = diagm(-1 => [sqrt(i) for i=1:N-1]) 
a⁻(N) = diagm(1 => [sqrt(i) for i=1:N-1]) 
n̄(N) = diagm(0 => [i for i=0:N-1]) 
δ(m, n) = m==n ? 1 : 0

function onehot(i::Int, N::Int, T::Type=Float64)
	v = zeros(T, N)
	v[i] = 1
	return v
end

function onehot(i::Int, j::Int, N::Int, T::Type=Float64)
	v = zeros(T, N, N)
	v[i,j] = 1
	return v
end

abstract type PropagationParas{T} end
Base.@kwdef struct rk45{T} <: PropagationParas{T}
    L::Int = 4
    a::Vector{T} = T.([1/6., 1/3., 1/3., 1/6.]) 
    b::Vector{T} = T.([1/2., 1/2., 1., 0.])
end
Base.@kwdef struct expo{T} <: PropagationParas{T}
    L::Int = 4
    a::Vector{T} = T.([1/1., 1/2., 1/3., 1/4.]) 
    b::Vector{T} = T.([1/1., 1/2., 1/3., 1/4.])
end

struct Times{T}
    δt::T
    Nt::Int
    t::Vector{T}
    P::PropagationParas{T}
    function Times(δt::T, Nt::Int; P::PropagationParas{T}=rk45{T}()) where {T}
        t = collect(1:Nt)*δt
        new{T}(δt, Nt, t, P)
    end
end

function Partialtr(A::Matrix{T}, newsize::Dims{N}, iᵣ::Int) where {T,N}
    A = reshape(A, newsize..., newsize...)
    ds = circshift(collect(1:N), -(iᵣ-1)) 
    ds = [ds; ds.+N]
    A = permutedims(A, ds)
    return A
end

function Base.:+(A::Matrix, B::Nothing)
    return A
end

function Base.:+(A::Nothing, B::Matrix)
    return B
end

function Base.:+(A::Nothing, B::Nothing)
    return nothing
end

function Base.:-(A::Matrix, B::Nothing)
    return A
end

function Base.:-(A::Nothing, B::Matrix)
    return -B
end

function Base.:-(A::Nothing, B::Nothing)
    return nothing
end


function Base.:*(A::Matrix, B::Nothing)
    return nothing
end

function Base.:*(A::Nothing, B::Matrix)
    return nothing
end

function Base.:*(A::Number, B::Nothing)
    return nothing
end

function Base.:*(A::Nothing, B::Number)
    return nothing
end

function LinearAlgebra.tr(::Nothing)
    return 0
end
