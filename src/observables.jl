getRDOr(A::Matrix{T}, ::Nothing=nothing) where {T} = real.(A) 
getRDOi(A::Matrix{T}, ::Nothing=nothing) where {T} = imag.(A) 
getPopulation(A::Matrix{T}, ::Nothing=nothing) where{T} = diag(real.(A))
getNorm(A::Matrix{T}, ::Nothing=nothing) where{T} = real.(tr(A))
getSz(A::Matrix{T}, ::Nothing=nothing) where{T} = real(tr(sz*A))
getProj(A::Matrix{T1}, B::Matrix{T2}) where {T1, T2} = real(tr(A*B)) 
getFlux(A::Matrix{T1}, B::Matrix{T2}) where {T1, T2} = real(tr(A*B)) 

function Observables(ρ::Vector, model::ModelADO, obs::Vector{Symbol})
    vals = Dict()
    for key in obs
        get_ob = Symbol(:get, key)
        val = [@eval $(get_ob)($(ρ[1]), $(try getfield(M, key) catch; nothing end)) for (iₛ, M) in model.System]
        vals[key] = val
    end
    return vals
end

"""
    getMPSElement(A, state)

Return the element of the MPS `A` for a given physical state `state`.

# Examples

```jldoctest
julia> A = randmps(6, 2, 10);

julia> elementmps(A, [1, 2, 1, 1, 1, 1])
```
"""
function getMPSElement(A, state)
    N = length(A)
    length(state) == N || throw(ArgumentError("indices do not match MPS"))
    ρ = A[1][:,:, state[1]]
    for i = 2:N
        ρ *= A[i][:,:, state[i]]
    end
    return ρ[1,1]
end

"""
    getReducedMPS(A::Vector, tags::Vector{:Symbol})

Return the reduced matrix product state by tracing out the environmental sites.

"""
function getReducedMPS(A::Vector, tags::Vector{Symbol}, i₀::Int=0)
    N = length(A)
    length(tags) == N || throw(ArgumentError("indices do not match MPS"))
    R = []
    M = fill(1.0, (1,1))
    for i = 1:N
        if tags[i] == :E
	    iᵦ = i == i₀ ? 2 : 1
            M *= A[i][:,:,iᵦ]
        else
            @tensor G[:] := M[-1,1] * A[i][1,-2,-3]
            push!(R, G)
            Dₗ, Dᵣ, d = size(A[i])
            M = Matrix(I, Dᵣ, Dᵣ)
        end
    end
    Nₛ = count(tags .!= :E)
    if findlast(tags .!= :E) != N
        @tensor G[:] := R[Nₛ][-1,1,-3] * M[1,-2] 
        R[Nₛ] = copy(G)
    end
    return R
end

function getRDONorm(II::Vector, A::Vector)
    length(II) == length(A) || throw(DimensionMismatch("The lengths of II and A are inconsistent"))
    ρ = fill(1.0, (1,1))
    for i in eachindex(II, A)
        ρ=rhoAB(ρ, II[i], A[i])
    end
    return ρ[1,1]
end

function getSingleSiteObservable(II::Vector, A::Vector, Ob::Matrix, i₀::Int)
    length(II) == length(A) || throw(DimensionMismatch("The lengths of II and A are inconsistent"))
    ρ = fill(1.0, (1,1))
    for i in eachindex(II, A)
        if i == i₀
            ρ=rhoAOB(ρ, II[i], A[i], Ob)
        else
            ρ=rhoAB(ρ, II[i], A[i])
        end
    end
    return ρ[1,1]
end

function getDoubleSiteObservable(II::Vector, A::Vector, Ob1::Matrix, Ob2::Matrix, i₀::Int, j₀::Int)
    length(II) == length(A) || throw(DimensionMismatch("The lengths of II and A are inconsistent"))
    if i₀ == j₀
	return getSingleSiteObservable(II, A, Ob1, i₀)
    else
        ρ = fill(1.0, (1,1))
        for i in eachindex(II, A)
            if i == i₀
                ρ=rhoAOB(ρ, II[i], A[i], Ob1)
            elseif i == j₀
                ρ=rhoAOB(ρ, II[i], A[i], Ob2)
            else
                ρ=rhoAB(ρ, II[i], A[i])
            end
        end
        return ρ[1,1]
    end
end

function getRDO(II::Vector, A::Vector, i₀::Int=1)
    length(II) == length(A) || throw(DimensionMismatch("The lengths of II and A are inconsistent"))
    ρₗ = fill(1.0, (1,1))
    ρᵣ = fill(1.0, (1,1))
    for i = 1:2i₀-2
        ρₗ=rhoAB(ρₗ, II[i], A[i])
    end
    for i = length(A):-1:2i₀+1
        ρᵣ=rhoAB(ρᵣ, II[i], A[i], :R)
    end
    @tensor RDS[a, s, s', c] := ρₗ[a₀,a] * A[2i₀-1][a₀, b₀, s] * A[2i₀][b₀, c₀, s'] * ρᵣ[c₀,c]
    return RDS[1, :, :, 1]
end

getObs(II::Vector, A::Vector, M::Matrix, i₀::Int=1) = real.(getSingleSiteObservable(II, A, M, i₀))
getObs(II::Vector, A::Vector, M1::Matrix, M2::Matrix, i₀::Int, j₀::Int) = getDoubleSiteObservable(II, A, M1, M2, i₀, j₀)
getObs(II::Vector, A::Vector, M::Matrix, i₀::Int, j₀::Int) = getDoubleSiteObservable(II, A, M, M, i₀, j₀)
getObs(II::Vector, A::Vector, M::Matrix, is::Vector{Int}) = [real.(getSingleSiteObservable(II, A, M, i₀)) for i₀ in is]
getObs(II::Vector, A::Vector, M::Vector{Matrix}, is::Vector{Int}) = [real.(getSingleSiteObservable(II, A, M[i₀], i₀)) for i₀ in is]
getPopulation(II::Vector, A::Vector, i₀::Int=1) = diag(real.(getRDO(II, A, i₀)))
function getWavepacket(II::Vector, A::Vector, U::Matrix, i₀::Int=1)
    RDM = getRDO(II, A, i₀)
    Pop = []
    for i=1:size(U)[1]
        push!(Pop, (U[i,:])'*RDM*U[i,:])
    end
    return real.(Pop)
end
getNorm(II::Vector, A::Vector) = real.(getRDONorm(II, A))

function Observables(A::Vector, model::ModelMPS, obs::Vector{Symbol})
    RDO = getReducedMPS(A, model.tags)
    vals = Dict()
    for key in obs
	if key == :Norm 
            val = getNorm(model.II, RDO)
	elseif key == :Population 
            val = [getPopulation(model.II, RDO, iₛ) for iₛ in keys(model.System)]
        else
	    val = [getObs(model.II, RDO, getfield(M, key), key == :Flux ? 2iₛ : 2iₛ-1) for (iₛ, M) in model.System]
        end
	vals[key] = val
    end
    return vals
end
