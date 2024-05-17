function initenvs(A::Vector, M::Vector, F::Nothing)
    N = length(A)
    F = Vector{Any}(undef, N+2)
    F[1] = fill!(similar(M[1],(1,1,1)), 1)
    F[N+2] = fill!(similar(M[1],(1,1,1)), 1)
    for k = N:-1:1
        F[k+1] = updaterightenv(A[k], M[k], F[k+2])
    end
    return F
end
initenvs(A::Vector, M::Vector) = initenvs(A, M, nothing)
initenvs(A::Vector, M::Vector, F::Vector) = F

function tdvp1sweep!(dt2, A::Vector, M::Vector, F=nothing; kwargs...)
    N = length(A)
    dt = dt2/2
    tol = get(kwargs, :tol, 1e-12)
    krylovdim = get(kwargs, :krylovdim, 30)
    maxiter = get(kwargs, :maxiter, 100)

    F = initenvs(A, M, F)
    AC = A[1]
    for k = 1:N-1
        AC, info = exponentiate(x->applyH1(x, M[k], F[k], F[k+2]), -im*dt, AC; tol=tol, krylovdim=krylovdim, maxiter=maxiter)
        A[k], C = QR(AC)
        F[k+1] = updateleftenv(A[k], M[k], F[k])
        C, info = exponentiate(x->applyH0(x, F[k+1], F[k+2]), im*dt, C; tol=tol, krylovdim=krylovdim, maxiter=maxiter)
        @tensor AC[:] := C[-1, 1] * A[k+1][1, -2, -3]
    end
    k = N
    AC, info = exponentiate(x->applyH1(x, M[k], F[k], F[k+2]), -im*dt2, AC; tol=tol, krylovdim=krylovdim, maxiter=maxiter)
    for k=N-1:-1:1
        C, A[k+1] = LQ(AC)
        F[k+2] = updaterightenv(A[k+1], M[k+1], F[k+3])
        C, info = exponentiate(x->applyH0(x, F[k+1], F[k+2]), im*dt, C; tol=tol, krylovdim=krylovdim, maxiter=maxiter)
        @tensor AC[:] := A[k][-1, 1, -3] * C[1, -2]
        AC, info = exponentiate(x->applyH1(x, M[k], F[k], F[k+2]), -im*dt, AC; tol=tol, krylovdim=krylovdim, maxiter=maxiter)
    end
    A[1] = AC
    return A, F
end

function QuantumStep!(ρ::Vector, vib::Vibs, model::ModelMPS, times::Times, F=nothing; kwargs...) 
    HMPO = DynamicalMPO(model, vib)
    tdvp1sweep!(times.δt, ρ, HMPO, F; kwargs...)
end

import Base.Threads.@threads
function HEOM(A::Vector, vib::Vibs, model::Model)
    δA = similar(A)
    Hₛ = model.Hₛ(vib.p..., vib.x...)
    op = similar(model.op, Matrix) 
    for I in eachindex(op, model.op)
        op[I] = model.op[I](vib.x...)
    end
    for node in model.bath 
        j = node.number
    	∑γ = 0. 
        for (jₘ, level) in enumerate(node.inds)
            (jₑ, jᵣ, jₚ) = model.bath.modes[jₘ]
            ∑γ += level * model.bath.γ[jᵣ][jₚ]
        end
        δA[j] = Hₛ * A[j] .- A[j] * Hₛ -im .* ∑γ * A[j]
    
        for (jₘ, j⁻) in node.top
            (jₑ, jᵣ, jₚ) = model.bath.modes[jₘ]
            δA[j] -= sqrt(node.inds[jₘ]) * model.Γν[jₑ, jᵣ] * ( model.bath.η[jᵣ][jₚ] * op[jᵣ] * A[j⁻] - conj(model.bath.η[jᵣ][jₚ]) * A[j⁻] * op[jᵣ])
        end
    
        for (jₘ, j⁺) in node.down
            (jₑ, jᵣ, jₚ) = model.bath.modes[jₘ]
            δA[j] -= sqrt(node.inds[jₘ]+1) * model.Γν[jₑ, jᵣ] * (op[jᵣ] * A[j⁺] - A[j⁺] * op[jᵣ])
        end
    end
    return δA
end

function QuantumEOM!(ρ::Vector, vib::Vibs, model::Model, times::Times)
    ρ_ = deepcopy(ρ)
    for I = 1:times.P.L
    	δρ = HEOM(ρ_, vib, model)
        if times.P isa rk45
    	    ρ_ .+= -im * times.δt * times.P.b[I] * δρ
        elseif times.P isa expo
            ρ_ = -im * times.δt * times.P.b[I] * δρ
        end
    	ρ .+= (-im * times.δt * times.P.a[I]) * δρ
    end
    for node in model.bath 
        i = node.number
	isa(ρ[i], Matrix) && (maximum(abs.(ρ[i])) < model.ϵ) && (ρ[i]=nothing)
    end
    return ρ
end

QuantumStep!(ρ::Vector, vib::Vibs, model::ModelADO, times::Times, F=nothing; kwargs...) = (QuantumEOM!(ρ, vib, model, times), nothing)

function XPEOM(A::Vector, vib::Vibs, model::ModelMPS)
    δvib = similar(vib)
    δvib.x .= vib.p ./ model.mass
    RDO = getReducedMPS(A, model.tags)
    for I in eachindex(δvib.p)
	for (iₑ, f) in model.∇Hₛ[I]
	    δvib.p[I] = -real(getSingleSiteObservable(model.II, RDO, f(vib.x...), iₑ))
	end
    end
    for jₘ in findall(x->x==:E, model.tags)
        RDO = getReducedMPS(A, model.tags, jₘ)
        (jₑ, jᵣ, jₚ) = model.modes[jₘ]
            for I in eachindex(δvib.p)
		for (iₑ, iᵣ, f) in model.∇op[I]
		    (iₑ == jₑ) && (iᵣ==jᵣ) && (δvib.p[I] -= real(getSingleSiteObservable(model.II, RDO, f(vib.x...), iₑ)))
		end
            end
    end
    return δvib
end

function XPEOM(A::Vector, vib::Vibs, model::ModelADO)
    δvib = similar(vib)
    δvib.x .= vib.p ./ model.mass
    for node in model.bath 
        j = node.number
	if node.tier==0
            for I in eachindex(δvib.p)
                δvib.p[I] = -real(tr(A[j].*model.∇Hₛ[I](vib.x...)))
            end
	elseif node.tier==1
            jₘ = findfirst(x->x==1, node.inds)
            (jₑ, jᵣ, jₚ) = model.bath.modes[jₘ]
            for I in eachindex(δvib.p)
                δvib.p[I] -= model.Γν[jₑ, jᵣ]*real(tr(A[j]*model.∇op[I][jᵣ](vib.x...)))
            end
        end
    end
    return δvib
end

function ClassicalStep!(vib::Vibs, ρ::Vector, model::Model, times::Times)
    vib_ = deepcopy(vib)
    for I = 1:times.P.L
    	δvib = XPEOM(ρ, vib_, model)
    	vib += times.δt * times.P.a[I] * δvib
        if times.P isa rk45
    	    vib_ += times.δt * times.P.b[I] * δvib
        elseif times.P isa expo
            vib_ = times.δt * times.P.b[I] * δvib
        end
    end
    return vib
end

function run_trajectory!(obs::Vector{Symbol}, vib::Vibs, model::Model, times::Times; verbose::Bool=false, kwargs...)
    vals = Dict{Symbol, Vector}()
    verbose && (vals[:time] = [])
    for key in obs
	vals[key] = []
    end

    ρ = deepcopy(model.ρ₀)
    F = nothing
    for iₜ = 1:times.Nt
        if verbose
            (ρ, F), ts, bytes, gctime, memallocs = @timed QuantumStep!(ρ, vib, model, times, F; kwargs...)
	    push!(vals[:time], ts)
        else
            ρ, F = QuantumStep!(ρ, vib, model, times, F; kwargs...)
        end
        vib = ClassicalStep!(vib, ρ, model, times)

        val = Observables(ρ, model, obs)
        for key in obs
	    push!(vals[key], val[key])
        end
    end
    return vals
end
