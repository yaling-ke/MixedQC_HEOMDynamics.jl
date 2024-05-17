module HEOMDynamics
    using LinearAlgebra, ITensors, TensorOperations, KrylovKit, JLD2, Printf, ArgParse, MPI, OrderedCollections, Combinatorics, Unitful, UnitfulAtomic, DelimitedFiles, Parameters, Polynomials, SpecialPolynomials, Distributions
    using Random: Random, AbstractRNG, SamplerTrivial, Xoshiro

    include("fundamentals.jl")
    include("parameters.jl")
    include("DVR.jl")
    include("sampling.jl")
    include("bath.jl")
    include("models.jl")
    include("observables.jl")
    include("dynamics.jl")
    include("MPS.jl")

    function runDynamics(observables::Vector{Symbol}, model::Model, times::Times; kwargs...)
        saved = get(kwargs, :saved, "")
        verbose = get(kwargs, :verbose, false)
        Init = get(kwargs, :Init, true)
        ntraj = get(kwargs, :ntraj, 1)

        MPI.Init()
        comm = MPI.COMM_WORLD
        rank = MPI.Comm_rank(comm)
        mpisize = MPI.Comm_size(comm)

        vals = Dict{Symbol, Vector}()
        vals_parallel = Dict{Symbol, Vector}()
        verbose && (vals[:time] = zeros(times.Nt))
        val = Observables(model.ρ₀, model, observables)
        for key in observables
            vals[key] = repeat([zero.(val[key])], times.Nt)
            vals_parallel[key] = []
        end

        rng=Xoshiro(1234)
        ntraj_ = Int(ntraj/mpisize)
        ntraj = ntraj_*mpisize
        MPI.Barrier(comm)
	for i=1:ntraj_
            @printf("%i/%i \n", i, ntraj_)
            xp = rand(rng, model.vibdis)
            vals_ = run_trajectory!(observables, xp, model, times; verbose=verbose)
            verbose && (vals[:time] .+= vals_[:time]./ntraj)
            for key in observables
                vals[key] .+= vals_[key]./ntraj
            end
	end
	MPI.Barrier(comm)

        for key in observables
            for it = 1:times.Nt
                push!(vals_parallel[key], [MPI.Reduce(vals[key][it][iₛ], +, 0, comm) for iₛ in keys(model.System)])
            end
        end

	if rank==0
            core = "core$(saved).jld2"
    	    if isfile(core)
                ntrajₒ = load(core, "ntraj")
                valsₒ = load(core, "data")
                for key in observables
                   vals_parallel[key] = (valsₒ[key]*ntrajₒ+vals_parallel[key]*ntraj)/(ntrajₒ+ntraj)
                end
    	    else
                ntrajₒ = 0
    	    end
            jldsave(core, ntraj=ntraj+ntrajₒ, data=vals_parallel)
            if verbose
                save_t = open("times$(saved).txt", Init ? "w" : "a")
                for it = 1:times.Nt
                    @printf(save_t, "timestep = %i/%i takes %.5f seconds\n", it, times.Nt, vals[:time][it])
                end
                println(save_t, "simulation time(s): $(sum(vals[:time]))")
                close(save_t)
            end
            for key in observables
                savequantity = open("$(key)$(saved).txt", Init ? "w" : "a")
                for it = 1:times.Nt
                    time = au2SI(u"fs", times.t[it])
                    write(savequantity, string(time, "\t"), [string(s, "\t") for s in vals_parallel[key][it]]..., "\n")
                end
                close(savequantity)
            end
	end
    end

    export EffectiveBoseBath, EffectiveFermiBath, BoseBath    
    export DVR, SineDVR, SincDVR, HermiteDVR, PES
    export Units, au2SI, Parameters
    export svdtrunc, onehot, MPOtoVector, physdims
    export s⁺, s⁻, sup, sdn, sz, sx, a⁺, a⁻, n̄, δ, Times
    export Model, DoubleWell, Morse, Photon, ProtonTransferMPS, PolaritonMPS, ProtonTransferADO, PolaritonADO
    export runDynamics
end# module HEOMDynamics
