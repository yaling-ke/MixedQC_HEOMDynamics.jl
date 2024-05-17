using LinearAlgebra, ITensors
using Unitful, UnitfulAtomic

args = Parameters()
ntraj  = get(args, "ntraj", 1)
nbaths = get(args, "nbaths", 1)
npoles = get(args, "npoles", 1)
ntier = get(args, "ntier", 1)
Rep = get(args, "representation", :ADO)
SysownBath = get(args, "SysownBath", true)

δt = get(args, "dt", 1.0)
Nt = get(args, "Nts", 1)
Init = get(args, "Init", true)
observables = get(args, "observables", [])
verbose = get(args, "verbose", false)
ϵ = get(args, "filter", 0.0)
Dmax = get(args, "Dmax", 1)

β = 1/get(args, "temperature", 1.0) # inverse temperatere
mass  = get(args, "mass", 1.0)
xᵈʷ  = get(args, "EquilibriumPosition", 1.0)
Eᵈʷ = get(args, "DoubleWellBarrier", 1.0)
Ωₘ = get(args, "BathCutoffFrequency", 1.0)
λₘ = get(args, "MoleculeBathCoupling", 1.0) * Ωₘ * sqrt(Eᵈʷ)/xᵈʷ
Nₘ = get(args, "nbasisMolecule", 1)
R = get(args, "DoubleWellXmax", 1.0)
NDVR = get(args, "nDVR", 10)
ωᶜ = get(args, "CavityFrequency", 1.0)
ηᶜ = get(args, "LightMatterCoupling", 1.0)
Ωᶜ = get(args, "CavityLossFrequency", 1.0)
λᶜ = (ωᶜ^2+Ωᶜ^2)*(1.0-exp(-β*ωᶜ))/get(args, "CavityLossStrength", 1.0)/(4Ωᶜ)

times = Times(δt, Nt)
    
DW = DoubleWell(xᵈʷ, Eᵈʷ; Nₑ=Nₘ, λ=ωᶜ*ηᶜ^2+λₘ, mass=mass, R = R, NDVR = NDVR)
if Rep == :MPS
    ITensors.space(::SiteType"DW") = DW.N
    ITensors.op(::OpName"Hm", ::SiteType"DW") = DW.H 
    ITensors.op(::OpName"Cm", ::SiteType"DW") = DW.q 
    ITensors.op(::OpName"Im", ::SiteType"DW") = DW.I
    model = PolaritonMPS(DW, sqrt(λₘ), Ωₘ, sqrt(λᶜ), Ωᶜ, ωᶜ, ηᶜ, β, npoles, ntier, Dmax)
    saved = "_D_$(Dmax)_Nm_$(Nₘ)_Poles_$(npoles)_Tier_$(ntier)_dt_$(round(au2SI(u"fs",δt); digits=3))_wc_$(round(au2SI(Units["cm^-1"], ωᶜ);digits=3))"
elseif Rep == :ADO
    saved = "_Nm_$(Nₘ)_Poles_$(npoles)_Tier_$(ntier)_dt_$(round(au2SI(u"fs",δt); digits=3))_wc_$(round(au2SI(Units["cm^-1"], ωᶜ);digits=3))"
    model = PolaritonADO(DW, λₘ, Ωₘ, λᶜ, Ωᶜ, ωᶜ, ηᶜ, β, npoles, ntier, ϵ)
end

runDynamics(observables, model, times; saved=saved, verbose=verbose, Init=Init, ntraj=ntraj)

