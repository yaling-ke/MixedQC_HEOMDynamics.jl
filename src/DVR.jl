abstract type DVR{T} end

"""
        SineDVR(npts::Int, xmin::T, xmax::T; mass::T=1.0) where {T}
Sine function basis for non-periodic functions over an interval from `xmin` to `xmax` with `npts` points.
"""
struct SineDVR{T} <: DVR{T}
    npts::Int
    x::Vector{T}
    KineticE::Matrix{T}
    function SineDVR(npts::Int, xmin::T, xmax::T; mass::T=1.0) where {T}
        L = xmax - xmin
        δx = L/(npts+1)
        x = xmin .+ δx * collect(1:npts)
        i₁ = repeat(collect(1:npts), 1, npts)
        i₂ = repeat(collect(1:npts)', npts, 1)
        KineticE = @. (-1.0)^(i₁-i₂)  * (1.0 /(sin(π/(2*(npts+1)) * (i₁-i₂))^2) - 1.0 / sin(π/(2*(npts+1)) * (i₁+i₂))^2)
        KineticE[diagind(KineticE)] .= (2.0 * (npts+1)^2 + 1.0)/3 .- 1.0 ./sin.(π/(npts+1).*collect(1:npts)).^2
        KineticE .*= (π/L)^2/(4mass)
        new{T}(npts, x, KineticE)
    end
end

struct SincDVR{T} <: DVR{T}
    npts::Int
    x::Vector{T}
    KineticE::Matrix{T}
    BasisF::Function
    function SincDVR(npts::Int, L::T; x₀::T=0.0, mass::T=1.0) where {T}
        δx = L/npts
        x = x₀ - L/2 .+ δx * (collect(1:npts) .- 1.0/2)
        i₁ = repeat(collect(1:npts), 1, npts)
        i₂ = repeat(collect(1:npts)', npts, 1)
        KineticE = @. 2.0 * (-1.0)^(i₁-i₂) /((i₁-i₂)*δx)^2
        KineticE[diagind(KineticE)] .= (π/δx).^2 ./3.0
        KineticE .*= 1.0/(2mass)
        function BasisF(xᵢ)
            return sinc.((xᵢ .- x)./δx)/sqrt(δx)
        end
        new{T}(npts, x, KineticE, BasisF)
    end
end

"""
        HermiteDVR(npts::Int, xmax::T; x₀::T=0.0, mass::T=1.0) where {T}
Hermite function basis for non-periodic functions over an interval from `-xmax` to `xmax` with `npts` points.
"""
struct HermiteDVR{T} <: DVR{T}
    npts::Int
    x::Vector{T}
    KineticE::Matrix{T}
    function HermiteDVR(npts::Int; xmax::T=1.0, x₀::T=0.0, mass::T=1.0) where {T}
        x = real.(roots(basis(Hermite, npts)))
        γ = maximum(x)/xmax
        x .= x₀ .+ x./γ
        w = exp.(-x.^2)
        L = maximum(x) - minimum(x)
        i₁ = repeat(collect(1:npts), 1, npts)
        i₂ = repeat(collect(1:npts)', npts, 1)
        x₁ = repeat(x, 1, npts)
        x₂ = repeat(x', npts, 1)
        KineticE = @. 2.0 * (-1.0)^(i₁-i₂) /((x₁-x₂)^2)
        KineticE[diagind(KineticE)] .= (2npts+1 .- x.^2) ./3
        KineticE .*= 1.0/(2mass)*γ
        new{T}(npts, x, KineticE)
    end
end

PES(dvr::DVR, f::Function) = f.(dvr.x)
