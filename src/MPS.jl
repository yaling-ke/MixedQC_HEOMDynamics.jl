"""
    mpsrightnorm!(A:Vector, jq::Int=1)
Right orthogonalize MPS `A` up to site `jq`
"""
function mpsrightnorm!(A::Vector, jq::Int=1)
	N = length(A)
	for i = N:-1:jq+1
		Dₗ, Dᵣ, d = size(A[i])
		C, AR = lq(reshape(A[i], Dₗ, Dᵣ*d))
		A[i] = reshape(Matrix(AR), Dₗ, Dᵣ, d)
		@tensor AC[:] := A[i-1][-1, 1, -3] * C[1, -2]
		A[i-1] = AC
	end
end

"""
    productstatemps(physdims::Dims{N}, Dmax, T::Type{<:Number}=Float64, state = :Vacuum, mpsorthog = :Right)

Return an MPS representing a product state with local Hilbert space dimensions given by `physdims`.

By default all bond dimensions will be 1 since the state is a product state. However, to embed the product state in a manifold of greater bond dimension, `Dmax` can be set accordingly.

The individual states of the MPS sites can be provided by setting `state` to a list of column vectors. Setting `state = :Vaccum` will produce an MPS in the vacuum state (where the state of each site is represented by a column vector with a 1 in the first row and zeros elsewhere). Setting `state = :FullOccupy` will produce an MPS in which each site is fully occupied (i.e. a column vector with a 1 in the last row and zeros elsewhere). 

The argument `mpsorthog` can be used to set the gauge of the resulting MPS.
"""
function productstatemps(physdims::Dims{N}, Dmax, T::Type{<:Number}=Float64; state = :Vacuum, mpsorthog = :Right) where {N}

	bonddims = Vector{Int}(undef, N+1)
	bonddims[1] = 1
	bonddims[N+1] = 1
	for i = 2:N
		bonddims[i] = min(Dmax, bonddims[i-1]*physdims[i-1])
	end
	for i = N:-1:2
		bonddims[i] = min(bonddims[i], bonddims[i+1]*physdims[i])
	end

	if state == :Vacuum
		statelist = [onehot(1, physdims[i], T) for i=1:N]
	elseif state == :FullOccupy
		statelist = [onehot(physdims[i], physdims[i], T) for i=1:N]
	elseif typeof(state) <: Vector 
		length(state) == N || throw(ErrorException("state list has length $(length(state)) while MPS has $N sites"))
		statelist = state
	else
		throw(ErrorException("state input not recognized"))
	end

	As = Vector{Any}(undef, N)
	for i = 1:N
		d = physdims[i]
		Dₗ = bonddims[i]
		Dᵣ = bonddims[i+1]
		A = zeros(T, Dₗ, Dᵣ, d)
		if typeof(state) <: Vector 
                    if statelist[i][1] == :SL
                        D = min(Dᵣ, size(statelist[i][2])[1])
                        A[1,1:D,:] = statelist[i][2][1:D,:]
                    elseif statelist[i][1] == :SR
                        D = min(Dₗ, size(statelist[i][2])[1])
                        A[1:D,1,:] = statelist[i][2][1:D,:]
                    else
                        A[1,1,:] = statelist[i][2]
                    end
		else
		    for j = 1:min(Dₗ, Dᵣ)
		    	A[j,j,:] = statelist[i]
		    end
		end
		As[i] = A
	end
	mpsrightnorm!(As)
	return As
end
prodcutstatemps(N::Int, d::Int, Dmax=1, T=Float64; state = :Vacuum, mpsorthog = :Right) = productstatemps(ntuple(n->d, N), Dmax, T; state=state, mpsorthog=mpsorthog)

"""
   U, S, Vt = svdtrunc(A;, truncdim = max(size(A)...), truncerr = 0)

Perform a truncated SVD, with maximum number of singular values to be `truncdim` or discarding any singular values smaller than `truncerr`. If both arguments are provided, the smaller number of singular values will be kept. 
"""
function svdtrunc(A; truncdim = max(size(A)...), truncerr=0)
    F = svd(A)
    d = min(truncdim, count(F.S .>= truncerr))
    return F.U[:,1:d], diagm(F.S[1:d]), F.Vt[1:d,:]
end

function physdims(M::Vector)
    N = length(M)
    res = Vector{Int}(undef, N)
    for (i, Mᵢ) in enumerate(M)
        res[i] = size(Mᵢ)[end]
    end
    return Dims(res)
end

function bonddims(M::Vector)
    N = length(M)
    res = Vector{Int}(undef, N+1)
    res[1] = res[N+1] = 1
    for i=1:N-1
        res[i+1] =size(M[i])[2]
    end
    return Dims(res)
end

function QR(A::AbstractArray; SVD=false)
    Dₗ, Dᵣ, d = size(A)
    if !SVD
        Q, R = qr(reshape(permutedims(A,[1,3,2]), Dₗ*d, Dᵣ))
        AL = permutedims(reshape(Matrix(Q), Dₗ, d, Dᵣ), [1,3,2])
        return AL, R
    else
        F = svd(reshape(permutedims(A,[1,3,2]), Dₗ*d, Dᵣ))
        AL = permutedims(reshape(F.U, Dₗ, d, Dᵣ), [1, 3, 2])
        R =  Diagonal(F.S) * F.Vt
        return AL, R
     end
end

function LQ(A::AbstractArray; SVD=false)
    Dₗ, Dᵣ, d = size(A)
    if !SVD
        L, Q = lq(reshape(A, Dₗ, Dᵣ*d))
        AR = reshape(Matrix(Q), Dₗ, Dᵣ, d)
        return L, AR
    else
        F = svd(reshape(A, Dₗ, Dᵣ*d))
        AR = reshape(F.Vt, Dₗ, Dᵣ, d)
        L =  F.U * Diagonal(F.S) 
        return L, AR
     end
end

function MPOtoVector(mpo::MPO)
	N = length(mpo)
	H = [Array(mpo[i], inds(mpo[i])...) for i=1:N]
	dims = size(H[1])
	H[1] = reshape(H[1], 1, dims...)
	dims = size(H[N])
	H[N] = reshape(H[N], dims[1], 1, dims[2], dims[3])
	return H
end

"""
    function rhoAAstar(ρ::Array{T1, 2}, A::Array{T2, 3}) where {T1, T2}
return 
```
  a0––A*––a         a
 /    |            /    
ρ     s      ===> ρ         
 \\    |            \\    
  b0––A––b          b
```
"""
function rhoAAstar(ρ::Array{T1, 2}, A::Array{T2, 3}) where {T1, T2}
    return @tensoropt ρ0[a,b] := ρ[a0,b0] * conj(A[a0, a, s]) * A[b0, b, s]
end

"""
    function rhoAOB(ρ::Array{T1, 2}, A::Array{T2, 3}, B::Array{T3, 3}, O::Array{T4, 2}) where {T1, T2, T3, T4}
return 
```
    a0––B––a
   /    |    
  /     s'   
 /      |              
ρ       O                   
 \\      |    
  \\     s    
   \\    |    
    b0––A ––b
```
"""
function rhoAOB(ρ::Array{T1, 2}, A::Array{T2, 3}, B::Array{T3, 3}, O::Array{T4, 2}) where {T1, T2, T3, T4}
    return @tensoropt ρ0[a,b] := ρ[a0,b0] * B[a0, a, s'] * O[s', s] * A[b0, b, s]
end

"""
    function rhoAOAstar(ρ::Array{T1, 2}, A::Array{T2, 3}, O::Array{T3,2}) where {T1, T2, T3}
return 
```
    a0––A*––a 
   /    |    
  /     s'   
 /      |              
ρ       O                   
 \\      |    
  \\     s    
   \\    |    
    b0––A ––b 
```
"""
function rhoAOAstar(ρ::Array{T1, 2}, A::Array{T2, 3}, O::Array{T3, 2}) where {T1, T2, T3}
    return @tensoropt ρ0[a,b] := ρ[a0,b0] * conj(A[a0, a, s']) * O[s', s] * A[b0, b, s]
end

"""
    function rhoAB(ρ::Array{T1, 2}, A::Array{T2, 3}, B::Array{T3, 3}, dir=:L) where {T1, T2, T3}
return 
```
    :L                :R
  a0––B-a          a––B-a0
 /    |                |   \\   
ρ     s      or        s    ρ    
 \\    |               |   /
  b0––A-b          b––A-b0
```
"""
function rhoAB(ρ::Array{T1, 2}, A::Array{T2, 3}, B::Array{T3, 3}, dir=:L) where {T1, T2, T3}
    if dir == :L
        return @tensoropt ρ0[a, b] :=  ρ[a0,b0] * B[a0, a, s] * A[b0, b, s]
    elseif dir == :R
        return @tensoropt ρ0[a, b] :=  ρ[a0,b0] * B[a, a0, s] * A[b, b0, s]
    end
end

"""
    function updateleftenv(A::Array{T1,3}, M::Array{T2,4}, FL) where {T1,T2}
return 
```
    a0––A*––a          a
   /    |             / 
  /     s'           /  
 /      |           /        
FL––b0––M ––b ===> F––b                   
 \\      |           \\     
  \\     s            \\    
   \\    |             \\   
    c0––A ––c          c
```
"""
function updateleftenv(A::Array{T1,3}, M::Array{T2,4}, FL) where {T1,T2}
    return @tensor F[a,b,c] := FL[a0,b0,c0] * conj(A[a0,a,s']) * M[b0,b,s',s] * A[c0,c,s]
end

"""
    function updatrightenv(A::Array{T1,3}, M::Array{T2,4}, FR) where {T1,T2}
return 
```
a––A*––a0          a
   |    \\           \\ 
   s'    \\           \\
   |      \\           \\        
b––M ––b0––FR ===>  b––F                   
   |       /           /      
   s      /           /       
   |     /           /
c––A ––c0           c
```
"""
function updaterightenv(A::Array{T1,3}, M::Array{T2,4}, FR) where {T1,T2}
    return @tensor F[a,b,c] := FR[a0,b0,c0] * conj(A[a,a0,s']) * M[b,b0,s',s] * A[c,c0,s]
end

"""
    applyH0(C,FL,FR)
return
```
    α      β          
   /        \\          
  /          \\         
 /            \\                
FL––––– a –––––FR ===>  α––HC––β                 
 \\            /             
  \\          /              
   \\        /         
    α'––C––β'          
```
"""
function applyH0(C, FL, FR)
    return @tensor HC[α,β] := FL[α,a,α'] * C[α',β'] * FR[β,a,β']
end

"""
    applyH1(AC,M,FL,FR)
return
```
    a0     a1         
   /        \\          
  /     s    \\              s
 /      |     \\             |   
FL––b0––M––b1––FR ===>  a0––HC––a1                
 \\      |      /             
  \\     s'    /              
   \\    |    /         
    c0––AC––c1          
```
"""
function applyH1(AC, M, FL, FR)
    return @tensoropt HAC[a0,a1,s] := FL[a0,b0,c0] * AC[c0,c1,s'] * M[b0,b1,s,s'] * FR[a1,b1,c1]
end

