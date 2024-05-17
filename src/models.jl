abstract type Model{T} end
abstract type Potential{T} end 

struct DoubleWellPotential{T} <: Potential{T} 
    f::Function
    ∇f::Function
    ω₀::T
    mass::T
    function DoubleWellPotential(Eᵇ::T, x₀::T, c::T=0.0; mass::T=1.0, λ::T=0.0) where {T} 
        f(x) = Eᵇ*((x/x₀)^2-1)^2 - c*(x/x₀)^3 + λ*x^2
        ∇f(x) = 4Eᵇ*x/x₀^2*((x/x₀)^2-1) - 3c/x₀*(x/x₀)^2 + 2λ*x
	ω₀ = sqrt(2Eᵇ/mass) * 2 / x₀
        new{T}(f, ∇f, ω₀, mass)
    end
end

struct MorsePotential{T} <: Potential{T}
    f::Function
    ∇f::Function
    ω₀::T
    mass::T
    function MorsePotential(Dₑ::T, a::T, xₑ::T; mass::T=1.0, E::T=0.) where {T}
        f(x) = Dₑ*(1 - exp(- a*(x - xₑ)))^2 + E
        ∇f(x) = 2a*Dₑ*(1 - exp(- a*(x - xₑ))) * exp(- a*(x - xₑ))
	ω₀ = a * sqrt(2*Dₑ/mass)
        new{T}(f, ∇f, ω₀, mass)
    end
end

struct DoubleWell{T}
    N::Int
    dw::DoubleWellPotential{T}
    dvr::DVR{T}
    H::Matrix{T}
    eigvals::Vector{T}
    eigvecs::Matrix{T}
    q::Matrix{T}
    I::Matrix{T}
    Proj::Matrix{T}
    Flux::Matrix{ComplexF64}
    function DoubleWell(x₀::T, Eᵇ::T, c::T=0.0; Nₑ::Int=10, λ::T=0.0, mass::T=1.0, R::T=100.0, NDVR::Int=500) where {T}
        dw = DoubleWellPotential(Eᵇ, x₀, c; mass=mass, λ=λ) 
        dvr = SineDVR(NDVR, -R, R; mass=mass)
        H = dvr.KineticE .+ diagm(PES(dvr, dw.f)) 
        eigvals, eigvecs = eigen(H)
        eigvals = eigvals[1:Nₑ]
        eigvecs = eigvecs[:,1:Nₑ]
        H = diagm(eigvals)
        q = eigvecs' * diagm(dvr.x) * eigvecs
        Iₑ = Matrix(I, Nₑ, Nₑ)
        Proj = eigvecs' * diagm(T[x >=0 for x in dvr.x]) * eigvecs
        Flux = im * (H*Proj - Proj*H)
        new{T}(Nₑ, dw, dvr, H, eigvals, eigvecs, q, Iₑ, Proj, Flux)
    end
end

struct Photon{T}
    n::Int
    H::Matrix{T}
    q::Matrix{T}
    Nbar::Matrix{T}
    function Photon(Nᶜ::Int, λᶜ::T, ωᶜ::T) where {T}
        q = (a⁺(Nᶜ) + a⁻(Nᶜ))/sqrt(2ωᶜ)
        H = diagm(0=> [ωᶜ*(i+0.5) for i=0:Nᶜ-1]) + λᶜ * q^2
        new{T}(Nᶜ, H, q, n̄(Nᶜ))
    end
end

abstract type ModelADO{T} <: Model{T} end

struct ProtonTransferADO{T} <: ModelADO{T}
    nsys::Int
    Hₛ::Array{T,2}
    op::Array{Matrix{T},2}
    Γν::Array{Float64}
    System::OrderedDict{Int, DoubleWell{T}}
    function ProtonTransferADO(model::DoubleWell{T}, λ::T) where {T}
        Γν = zeros(T, (1, 1))
        Γν[1,1] = sqrt(λ)
        op = Array{Matrix{T}, 2}(undef, (1, 1))
        op[1,1] = model.q
        new{T}(1, model.H, op, Γν, OrderedDict(1=>model))
    end
end

struct PolaritonADO{T} <: ModelADO{T}
    nsys::Int
    mass::Vector
    Hₛ::Function
    ∇Hₛ::Vector{Function}
    op::Vector{Function}
    ∇op::Vector{Vector{Function}}
    Γν::Array{Float64}
    System::OrderedDict{Int, DoubleWell{T}}
    vibdis::VibDistribution 
    bath::Bath
    ρ₀::Vector
    ϵ::Float64
    function PolaritonADO(model::DoubleWell{T}, λₘ::T, Ωₘ::T, λᶜ::T, Ωᶜ, ωᶜ::T, ηᶜ::T, β, npoles::Int=0, ntier::Int=2, ϵ::Float64=1e-12) where {T}
        Hᶜ(p,x) = p^2/2.0 + 1/2*ωᶜ^2*x^2 + λᶜ * x^2
	qᶜ(x) = x
        Hₛ(p,x) = model.H + Hᶜ(p,x) * model.I + sqrt(2ωᶜ^3)*ηᶜ* qᶜ(x) * model.q
        ∇Hₛ = Array{Function, 1}(undef, 1)
        ∇Hₛ[1] = x -> (ωᶜ^2+2λᶜ) * x * model.I + sqrt(2ωᶜ^3)*ηᶜ* model.q
        Γν = zeros(T, (2, 2))
        Γν[1,1] = sqrt(λᶜ)
        Γν[2,2] = sqrt(λₘ)
        op = Array{Function, 1}(undef, 2)
        op[1] = x -> qᶜ(x) * model.I
        op[2] = x -> model.q
        ∇op = Array{Vector{Function}, 1}(undef, 1)
        ∇op[1] = [x->model.I, x->0]

        σₓ = 1/sqrt(2*ωᶜ*tanh(ωᶜ*β/2))
        σₚ = 1/sqrt(2/ωᶜ*tanh(ωᶜ*β/2))
        vibdis = VibDistribution(0., σₓ, 0., σₚ, 1)
        
        bath = BoseBath(;β=β, Ω=[Ωᶜ, Ωₘ], nsys=1, nbaths=2, npoles=[0, npoles], Tier=ntier, SysownBath=true)

        ρₘ = exp(-β*model.H/2.0) * (model.I .- model.Proj) * exp(-β*model.H/2.0)
        ρ₀ = Vector(undef, bath.Nₕ)
        ρ₀[1] = ComplexF64.(ρₘ/tr(ρₘ))
        ρ₀[2:end] .= nothing

        new{T}(1, [1.0], Hₛ, ∇Hₛ, op, ∇op, Γν, OrderedDict(1=>model), vibdis, bath, ρ₀, ϵ)
    end
end

abstract type ModelMPS{T} <: Model{T} end

struct ProtonTransferMPS{T} <: ModelMPS{T}
    os::Sum
    sites::Vector{Index{Int64}}
    pdims::Dims
    tags::Vector{Symbol}
    II::Vector{Array{Float64}}
    System::OrderedDict{Int, DoubleWell{T}}
    function ProtonTransferMPS(dw::DoubleWell{T}, Γ, β, Ω, npoles::Int=0, ntier::Int=2) where {T}

        bath = EffectiveBoseBath(β, Ω, npoles+1) 
        os = OpSum()
        
        os += "H", 1
        os -= "H", 2
        
        i_pos = 2
        for iₚ = 1:bath.Nₚ
        	i_pos +=1
        	os += (-im*bath.γ[iₚ], "N", i_pos)
        	os += (Γ, "C", 1, "a", i_pos)
        	os += (-Γ, "C", 2, "a", i_pos)
        	os += (Γ*bath.η[iₚ], "C", 1, "a†", i_pos)
        	os += (-Γ*conj(bath.η[iₚ]), "C", 2, "a†", i_pos)
        end
        
        sys_sites = siteinds("DW",2)
        bath_sites = siteinds("Boson", dim=ntier, (npoles+1))
        sites=[sys_sites..., bath_sites...]
        tags = [:S, :S, repeat([:E], npoles+1)...]
        A₁ = zeros(1, dw.N, dw.N)
        A₂ = zeros(dw.N, 1, dw.N)
        for iₛ=1:dw.N
            A₁[1,iₛ,:] = onehot(iₛ, dw.N)
            A₂[iₛ,1,:] = onehot(iₛ, dw.N)
        end
        II = []
        push!(II, A₁, A₂)
        new{T}(os, sites, Dims(dims(sites)), tags, II, OrderedDict(1=>dw))
    end
end

struct Cavity{T} <: ModelMPS{T}
    os::Sum
    sites::Vector{Index{Int64}}
    pdims::Dims
    tags::Vector{Symbol}
    II::Vector{Array{Float64}}
    System::OrderedDict{Int, Photon{T}}
    function Cavity(pt::Photon{T}, Γᶜ, Ωᶜ, β, npoles::Int=0, ntier::Int=2) where {T}

        bath = EffectiveBoseBath(β, Ωᶜ, npoles+1) 
        os = OpSum()

        os += "H", 1
        os -= "H", 2
        
        for iₚ = 1:bath.Nₚ
        	os += (-im*bath.γ[iₚ], "N", 2+iₚ)
        	os += (Γᶜ, "C", 1, "a", 2+iₚ)
        	os += (-Γᶜ, "C", 2, "a", 2+iₚ)
        	os += (Γᶜ*bath.η[iₚ], "C", 1, "a†", 2+iₚ)
        	os += (-Γᶜ*conj(bath.η[iₚ]), "C", 2, "a†", 2+iₚ)
        end
        
        pt_sites = siteinds("Photon",2)
        bath_sites = siteinds("Boson", dim=ntier, (npoles+1))
        sites=[pt_sites..., bath_sites...]
        tags = [:S, :S, repeat([:E], npoles+1)...]
        B₁ = zeros(1, pt.n, pt.n)
        B₂ = zeros(pt.n, 1, pt.n)
        for iₛ=1:pt.n
            B₁[1,iₛ,:] = onehot(iₛ, pt.n)
            B₂[iₛ,1,:] = onehot(iₛ, pt.n)
        end
        II = []
        push!(II, B₁, B₂)
        new{T}(os, sites, Dims(dims(sites)), tags, II, OrderedDict(1=>pt))
    end
end

struct PolaritonMPS{T} <: ModelMPS{T}
    os::Sum
    sites::Vector{Index{Int64}}
    pdims::Dims
    tags::Vector{Symbol}
    II::Vector{Array{Float64}}
    System::OrderedDict{Int, DoubleWell{T}}
    vibdis::VibDistribution 
    bathm::EffectiveBoseBath
    bathc::EffectiveBoseBath
    Hᶜ::Function
    qᶜ::Function
    ηᶜ::Float64
    ωᶜ::Float64
    Γᶜ::Float64
    modes::Dict
    ∇Hₛ::Vector{Vector}
    ∇op::Vector{Vector}
    mass::Vector
    ρ₀::Vector
    function PolaritonMPS(dw::DoubleWell{T}, Γₘ, Ωₘ, Γᶜ, Ωᶜ, ωᶜ, ηᶜ, β, npoles::Int=0, ntier::Int=2, Dmax::Int=10) where {T}

        bathm = EffectiveBoseBath(β, Ωₘ, npoles+1) 
        bathc = EffectiveBoseBath(β, Ωᶜ, 1) 

	modes = Dict()
        os = OpSum()
        
        for iₚ = bathm.Nₚ:-1:1
            os += (-im*bathm.γ[iₚ], "N", bathm.Nₚ+1-iₚ)
            os += (Γₘ, "a", bathm.Nₚ+1-iₚ, "Cm", bathm.Nₚ+1)
            os += (-Γₘ, "a", bathm.Nₚ+1-iₚ, "Cm", bathm.Nₚ+2)
            os += (Γₘ*bathm.η[iₚ], "a†", bathm.Nₚ+1-iₚ, "Cm", bathm.Nₚ+1)
            os += (-Γₘ*conj(bathm.η[iₚ]), "a†", bathm.Nₚ+1-iₚ, "Cm", bathm.Nₚ+2)
            modes[bathm.Nₚ+1-iₚ]=(1, 1, iₚ)
        end

        os += "Hm", bathm.Nₚ+1
        os -= "Hm", bathm.Nₚ+2

        for iₚ = 1:bathc.Nₚ
            os += (-im*bathc.γ[1,iₚ], "N", bathm.Nₚ+2+iₚ)
            modes[bathm.Nₚ+2+iₚ]=(1, 2, iₚ)
        end
        
        dw_sites = siteinds("DW",2)
        bathm_sites = siteinds("Boson", dim=ntier, bathm.Nₚ)
        bathc_sites = siteinds("Boson", dim=ntier, bathc.Nₚ)
        sites=[bathm_sites..., dw_sites..., bathc_sites...]
        tags = [repeat([:E], bathm.Nₚ)..., :S, :S, repeat([:E], bathc.Nₚ)...]
        A₁ = zeros(1, dw.N, dw.N)
        A₂ = zeros(dw.N, 1, dw.N)
        for iₛ=1:dw.N
            A₁[1,iₛ,:] = onehot(iₛ, dw.N)
            A₂[iₛ,1,:] = onehot(iₛ, dw.N)
        end
        II = []
        push!(II, A₁, A₂)

	σₓ = 1/sqrt(2*ωᶜ*tanh(ωᶜ*β/2))
	σₚ = 1/sqrt(2/ωᶜ*tanh(ωᶜ*β/2))
	vibdis = VibDistribution(0., σₓ, 0., σₚ, 1)

        Hᶜ(p,x) = p^2/2 + 1/2*ωᶜ^2*x^2 + Γᶜ^2 * x^2
	qᶜ(x) = x

        ∇Hₛ = Array{Vector, 1}(undef, 1)
        ∇Hₛ[1] = [(1, x -> (ωᶜ^2+2Γᶜ^2) * x * dw.I + sqrt(2ωᶜ^3)*ηᶜ* dw.q)]
        ∇op = Array{Vector, 1}(undef, 1)
        ∇op[1] = [(1, 2, x->Γᶜ*dw.I)]

        pdims = Dims(dims(sites))
        state = []
        for iₚ=1:bathm.Nₚ
            push!(state, (:E, onehot(1, pdims[iₚ])))
        end
        ρₘ = exp(-β*dw.H/2.0) * (dw.I .- dw.Proj) * exp(-β*dw.H/2.0)
        U, S, Vt = svdtrunc(ρₘ/tr(ρₘ))
        push!(state, (:SL, transpose(U*sqrt.(S))))
        push!(state, (:SR, sqrt.(S)*Vt))
        for iₚ=1:bathc.Nₚ
            push!(state, (:E, onehot(1, pdims[bathm.Nₚ+2+iₚ])))
        end
        ρ₀ = productstatemps(pdims, Dmax; state=state)
        save_info = open("bonddimensions.txt", "w")
        println(save_info, "bond dimensions: ", bonddims(ρ₀))
        close(save_info)


        new{T}(os, sites, pdims, tags, II, OrderedDict(1=>dw), vibdis, bathm, bathc, Hᶜ, qᶜ, ηᶜ, ωᶜ, Γᶜ, modes, ∇Hₛ, ∇op, [1.0], ρ₀)
    end
end

function DynamicalMPO(model::Model, vib::Union{T, Vector{T}, Vibs}) where {T}
    os = copy(model.os)
    SuperHamiltonian=MPO(os, model.sites)
    HamiltonianMPO = MPOtoVector(SuperHamiltonian)
    return HamiltonianMPO
end

function DynamicalMPO(model::PolaritonMPS, vib::Vibs)
    os = copy(model.os)

    @unpack bathm, bathc, Hᶜ, qᶜ, ηᶜ, ωᶜ, Γᶜ = model
    os += Hᶜ(vib.p..., vib.x...), "Im", bathm.Nₚ+1
    os -= Hᶜ(vib.p..., vib.x...), "Im", bathm.Nₚ+2
    os += (sqrt(2ωᶜ^3)*ηᶜ*qᶜ(vib.x...), "Cm", bathm.Nₚ+1)
    os -= (sqrt(2ωᶜ^3)*ηᶜ*qᶜ(vib.x...), "Cm", bathm.Nₚ+2)

    for iₚ = 1:bathc.Nₚ
    	os += (Γᶜ*qᶜ(vib.x...), "a", bathm.Nₚ+2+iₚ)
    	os += (-Γᶜ*qᶜ(vib.x...), "a", bathm.Nₚ+2+iₚ)
    	os += (Γᶜ*bathc.η[iₚ]*qᶜ(vib.x...), "a†", bathm.Nₚ+2+iₚ)
    	os += (-Γᶜ*conj(bathc.η[iₚ])*qᶜ(vib.x...), "a†", bathm.Nₚ+2+iₚ)
    end

    SuperHamiltonian=MPO(os, model.sites)
    HamiltonianMPO = MPOtoVector(SuperHamiltonian)
    return HamiltonianMPO
end
