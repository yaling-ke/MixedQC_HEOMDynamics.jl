function PadeBose(N)
    B₊(m) = 2m+1
    A₊ = [(δ(m, n+1)+δ(m,n-1))/sqrt(B₊(m) * B₊(n)) for m = 1:2N, n = 1:2N]
    Am₊ = [(δ(m, n+1)+δ(m,n-1))/sqrt(B₊(m+1) * B₊(n+1)) for m = 1:2N-1, n = 1:2N-1]
    ξ = sort(2. ./ filter(x -> x>0, eigen(A₊).values))
    χ = sort(2. ./ filter(x -> x>1e-13, eigen(Am₊).values))
    κ = fill(0.5*N*B₊(N+1), N)
    for i in 1:N
    	for j = 1:N-1
    	    κ[i] *=  (j==i ?  χ[j]^2-ξ[i]^2 : (χ[j]^2-ξ[i]^2)/(ξ[j]^2-ξ[i]^2)) 
    	end
    	κ[i] *=  (i==N ?  1 : 1 /(ξ[end]^2-ξ[i]^2))
    end
    return ξ, κ
end

nb(β, E) = 1 / (1 - exp(-β*E))
nb(β, E, ξ, κ) =  reduce(+, 2*β*E .* κ ./ ((β*E)^2 .+ ξ .^2); init = 0.5+1/(β*E))

"""
    function EffectiveBoseBath(β, Ω, Nₚ::Int=0; HighTemp=false, Matsubara=false) 

Return  a struct EffectiveBoseBath  with the fields: `Nₚ`, `η`, and `γ`. 
`Nₚ` is the number of poles per bath, and the effective bath parameters are coefficients `η` and exponents `γ`.  `β` is the inverse temperature and  `Ω` is the bath broadening width.
"""
struct EffectiveBoseBath{T1, T2}
    Nₚ::Int
    η::Vector{T1}
    γ::Vector{T2}
    function EffectiveBoseBath(β::Number, Ω::Number, Nₚ::Int=0; HighTemp=false, Matsubara=false) 
        η = zeros(ComplexF64, Nₚ)
        γ = zeros(Float64, Nₚ)
    
        γ[1] = Ω
        η[1] = (HighTemp ? Ω*(2.0/(β*Ω) - im) : Ω*(cot(Ω*β/2) - im))
        if Matsubara
            ξ = 2π.*collect(1:Nₚ-1)
            κ = ones(Nₚ-1)
        else
            ξ, κ = PadeBose(Nₚ-1)
        end
        γ[2:Nₚ] = [ξ[iₚ]/β for iₚ=1:Nₚ-1]
        η[2:Nₚ] = [4κ[iₚ]*Ω*γ[iₚ+1]/β/(γ[iₚ+1]^2 - Ω^2) for iₚ=1:Nₚ-1]
        new{ComplexF64, Float64}(Nₚ, η, γ)
    end
end

function CoeffsBose(β, Ω, poles::Vector; HighTemp=false, Matsubara=false) 
    Nₗ = length(poles)
    isa(β, Number) && (β = repeat([β], Nₗ))
    isa(Ω, Number) && (Ω = repeat([Ω], Nₗ))
    
    γ = []
    η = [] 
    for (i₁, pole) in enumerate(poles)
        b = EffectiveBoseBath(β[i₁], Ω[i₁], length(pole); HighTemp=HighTemp, Matsubara=Matsubara)
	push!(γ, b.γ)
	push!(η, b.η)
    end
    return η, γ
end

abstract type Node end
abstract type Hierarchy{T<:Node} end

mutable struct BoseNode <: Node
    tier::Int
    number::Int
    inds::Vector{Int}
    top::Dict{Int64, Int64}
    down::Dict{Int64, Int64}
end

function indices(iₘ, MaxTier, Tier, Nₘ)
    if iₘ < Nₘ
    	iₘ += 1
    	inds = []
    	for iooc = 0:min(MaxTier, Tier[iₘ-1])
    	    for elem in indices(iₘ, MaxTier-iooc, Tier, Nₘ)
    	    	push!(inds, [iooc, elem...])
    	    end
    	end
    	return inds
    elseif iₘ == Nₘ
    	return collect(0:min(MaxTier, Tier[iₘ]))
    end
end

struct BoseHierarchy{T<:Node} <: Hierarchy{T}
    nodes::Vector{T}
    function BoseHierarchy(Nₘ, TruncationTier)
    	tree = []
    	number = 0
	MaxTier = maximum(TruncationTier)
    	for elem in indices(1, MaxTier, TruncationTier, Nₘ)
            push!(tree, BoseNode(sum(elem), number += 1, length(elem) == 1 ? [elem] : elem, Dict(), Dict()))
    	end
    
    	lowtier=(filter(x-> x.tier==0, tree))
    	for tier=1:MaxTier
    	    hightier=(filter(x-> x.tier==tier, tree))
    	    for (s1, s2) in Iterators.product(lowtier, hightier)
    	    	for iₘ = 1:Nₘ
    	    	    if iₘ == 1
    	    	    	match = ((s1.inds[1] == s2.inds[1]-1) && isequal(s1.inds[2:end], s2.inds[2:end])) 
    	    	    elseif iₘ == Nₘ
    	    	    	match= (isequal(s1.inds[1:Nₘ-1], s2.inds[1:Nₘ-1]) && (s1.inds[Nₘ] == s2.inds[Nₘ]-1))
    	    	    else
    	    	    	match = (isequal(s1.inds[1:iₘ-1], s2.inds[1:iₘ-1]) && (s1.inds[iₘ] == s2.inds[iₘ]-1) && isequal(s1.inds[iₘ+1:end], s2.inds[iₘ+1:end]))
    	    	    end
    	    	    if match
    	    	    	push!(s1.down, iₘ => s2.number)
    	    	    	push!(s2.top, iₘ => s1.number)
    	    	    end
    	    	end
    	    end
    	    lowtier = hightier
    	end
        new{BoseNode}(tree)
    end
end

Base.length(iter::Hierarchy) = length(iter.nodes)
Base.iterate(iter::Hierarchy) = iterate(iter.nodes)
Base.iterate(iter::Hierarchy, state) = iterate(iter.nodes, state)
Base.getindex(iter::Hierarchy, i::Union{Int, Vector{Int}, UnitRange}) = getindex(iter.nodes, i)
Base.firstindex(iter::Hierarchy) = 1

abstract type Bath end

struct BoseBath <: Bath
    nbaths::Int
    npoles::Union{Int, Vector{Int}}
    modes::Array{Tuple}
    hierarchy::BoseHierarchy
    Nₕ::Int
    η::Vector
    γ::Vector
    function BoseBath(;β=1., Ω=1., Γ=1., nsys::Int=1, nbaths::Int=1, SysownBath::Bool=false, npoles::Union{Int, Vector{Int}}=0, Tier::Union{Int, Vector{Int}}=0, SpectralDensity::Symbol=:Debye)
	Nₚ = isa(npoles, Int) ? repeat([1:npoles+1], nbaths) : [1:p+1 for p in npoles]
	modes = []
	for (i, p) in enumerate(Nₚ)
            if SysownBath
                push!(modes, Iterators.product(i, i, p)...)
            else
                push!(modes, Iterators.product(1:nsys, i, p)...)
	    end
        end
        η,γ = CoeffsBose(β, Ω, Nₚ)
        hierarchy = BoseHierarchy(length(modes), isa(Tier, Int) ? repeat([Tier], length(modes)) : Tier)
        new(nbaths, npoles, modes, hierarchy, length(hierarchy), η, γ)
    end
end

Base.length(iter::Bath) = length(iter.hierarchy)
Base.iterate(iter::Bath) = iterate(iter.hierarchy)
Base.iterate(iter::Bath, state) = iterate(iter.hierarchy,state)
Base.getindex(iter::Bath, i::Union{Int, Vector{Int}, UnitRange}) = getindex(iter.hierarchy, i)
Base.firstindex(iter::Bath) = 1
