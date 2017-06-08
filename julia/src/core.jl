#!/usr/bin/env julia

# Rockfish

using PyPlot
using StatsBase
abstract AbstractModel
using PyCall
using ProgressMeter
using LinearOperators
using IterativeSolvers
@pyimport healpy

type CoreModel
    flux::Vector{Vector{Float64}}  # Flux of model components
    noise::Vector{Float64}         # Noise flux
    systematics::LinearOperator    # Systematic (co-)variance
    exposure::Vector{Float64}      # Exposure
end

function infomatrix(model::CoreModel; solver = "direct")
    D = opDiagonal(model.noise./model.exposure) + model.systematics
    n = length(model.flux)
    x = Vector{Vector{Float64}}(n)
    if solver == "direct"
        invD = inv(full(D))
        for i in 1:n
            x[i] = invD*model.flux[i]
        end
    elseif solver == "cg"
        for i in 1:n
            Pl = opDiagonal(diag(D))
            x[i] = cg(D, model.flux[i], Pl = Pl, verbose=true, maxiter = 10)
        end
    else
        error("Solver unknown.")
    end
    I = Matrix{Float64}(n,n)
    for i in 1:n
        for j in 1:i
            tmp = sum(model.flux[i].*x[j])
            I[i,j] = tmp
            I[j,i] = tmp
        end
    end
    return I
end

function infoflux(model::CoreModel; solver = "direct", maxiter = 10000000)
    D = opDiagonal(model.noise./model.exposure) + model.systematics
    n = length(model.flux)
    x = Vector{Vector{Float64}}(n)
    if solver == "direct"
        invD = inv(full(D))
        for i in 1:n
            x[i] = invD*model.flux[i]
        end
    elseif solver == "cg"
        for i in 1:n
            x[i] = cg(D, model.flux[i], verbose=true, maxiter = maxiter)
        end
    else
        error("Solver unknown.")
    end
    I = Matrix{Float64}(n,n)
    F = Matrix{Vector{Float64}}(n,n)
    for i in 1:n
        for j in 1:i
            tmp = x[i].*x[j].*model.noise./model.exposure.^2
            F[i,j] = tmp
            F[j,i] = tmp
            tmp = sum(model.flux[i].*x[j])
            I[i,j] = tmp
            I[j,i] = tmp
        end
    end
    return F, I
end

function effective(I::Matrix{Float64}, i::Int64)
    invI = inv(I)
    return 1/invI[i,i]
end

function effective(F::Matrix{Vector{Float64}}, I::Matrix{Float64}, i::Int64)
    n = size(I, 1)
    if n == 1
        return F[i,i]
    end
    indices = setdiff(1:n, i)
    eff_F = F[i,i]
    C = Vector{Float64}(n-1)
    B = Matrix{Float64}(n-1,n-1)
    for j in 1:n-1
        C[j] = I[indices[j],i]
        for k in 1:n-1
            B[j,k] = I[indices[j], indices[k]]
        end
    end
    invB = inv(B)
    for j in 1:n-1
        for k in 1:n-1
            eff_F = eff_F - F[i,indices[j]]*invB[j,k]*C[k]
            eff_F = eff_F - C[j]*invB[j,k]*F[indices[k],i]
            for l in 1:n-1
                for m in 1:n-1
                    eff_F = eff_F + C[j]*invB[j,l]*F[indices[l],indices[m]]*invB[m,k]*C[k]
                end
            end
        end
    end
    return eff_F
end

function tensorproduct(Σ1::Matrix{Float64}, Σ2::Matrix{Float64})
    return tensorproduct(LinearOperator(Σ1), Σ2)
end

function tensorproduct(Σ1::LinearOperator, Σ2::LinearOperator)
    return tensorproduct(Σ1, full(Σ2))
end

function tensorproduct(Σ1::LinearOperator, Σ2::Matrix{Float64})
    n1 = size(Σ1,1)
    n2 = size(Σ2,1)
    N = n1*n2
    function Σ(x::Vector{Float64})
        A = reshape(x, (n1, n2))
        B = zeros(A)
        for i in 1:n2
            y = Σ1*A[:,i]
            for j in 1:n2
                B[:,j] += Σ2[i,j]*y
            end
        end
        return reshape(B, N)
    end
    return LinearOperator(N, N, true, true, x->Σ(x))
end

function Σ_hpx(nside::Int64; sigma::Float64 = 0., scale::Any = 1.)
    npix = healpy.nside2npix(nside)
    function hpxconvolve(x::Vector{Float64})
        if sigma != 0.
            alm = healpy.map2alm(x.*scale)
            x = healpy.alm2map(alm, nside, sigma = deg2rad(sigma), verbose=false)
            return x.*scale
        end
    end
    function flat(x::Vector{Float64})
        return scale*sum(x.*scale)
    end
    if sigma == Inf
        return LinearOperator(npix, npix, true, true, x->flat(x))
    else
        return LinearOperator(npix, npix, true, true, x->hpxconvolve(x))
    end
end

#type AdditiveModel <: AbstractModel
#    components::Vector{Array{Float64}}
#    variance::Array{Float64}
#    exposure::Array{Float64}
#    function AdditiveModel(comps)
#        variance = zeros(comps[1])
#        exposure = zeros(comps[1])
#        new(comps, variance, exposure)
#    end
#end
#
#function setvariance!(m::AbstractModel, variance::Array{Float64})
#    m.variance = variance
#
#end
#function setexposure!(m::AbstractModel, exposure::Float64)
#    m.exposure = ones(m.variance)*exposure
#end
#
#function fishermatrix(m::AdditiveModel, i::Int64, j::Int64)
#    return sum(m.components[i] .* m.components[j] ./ m.variance .* m.exposure)
#end
#
#function fishermatrix(m::AdditiveModel)
#    N = length(m.components)
#    I = zeros((N,N))
#    for i in 1:N
#        for j in 1:i
#            temp = fishermatrix(m, i, j)
#            I[i,j] = temp
#            I[j,i] = temp
#        end
#    end
#    return I
#end
#
#function informationflux(M::AdditiveModel, i::Int64)
#    N = length(M.components)
#    indices = setdiff(1:N, i)
#
#    F = M.components[i] .* M.components[i] ./ M.variance
#    I = fishermatrix(M)
#    I_B = I[indices, indices]
#    invI_B = inv(I_B)
#    for j in 1:N-1
#        for k in 1:N-1
#            F_C = M.components[i] .* M.components[indices[j]] ./ M.variance
#            F = F - 2*invI_B[j, k]*I[i, indices[k]]*F_C
#            for l in 1:N-1
#                for m in 1:N-1
#                    F_B = M.components[indices[l]] .* M.components[indices[m]] ./ M.variance
#                    F = F + I[i, indices[j]]*invI_B[j, l]*invI_B[m, k]*I[i, indices[k]]*F_B
#                end
#            end
#        end
#    end
#
#    return F
#end
#
## Instrument specific 
#
#function setradiobg!(t::AdditiveModel, Tsys::Float64, Δν::Float64, τ::Float64,
#                    G::Float64, ΩA::Float64)
#    total = ones(t.components[1])
#    total = ((G^2*ΩA^2/Tsys^2*Δν)^-1)*total
#    t.variance = total
#    t.exposure = τ
#end
#
#function setpoissonbg!(t::AdditiveModel, exposure::Float64, θ::Array{Float64,1})
#    variance = sum([θ[k]*t.components[k] for k in 1:length(t.components)])
#    t.variance = variance
#    t.exposure = exposure*ones(variance)
#end
#
#"Get upper limit in background-limited regime."
#function upperlimitBL(Sigma::Array{Float64}, i::Integer, alpha::Real)
#    Z = 1.64  # FIXME: Get from alpha
#    return Z*Sigma[i,i]^0.5
#end
#
#type SysMatrixModel <: AbstractModel
#    comp::Array{Float64,1}
#    variance::Array{Float64,1}
#    exposure::Array{Float64,1}
#    Sigma::LinearOperator
#    function SysMatrixModel(comp, bg, Sigma)
#        variance = bg
#        exposure = zeros(comp)
#        new(comp, variance, exposure, Sigma)
#    end
#end
#
#type PointModel <: AbstractModel
#    components::Tuple{Array{Array{Array{Float64}, 1}, 1}, Int}
#    variance::Array{Float64}
#    exposure::Array{Float64}
#end
#
##function informationflux(M::SysMatrixModel)
##    shape = size(M.exposure)
##    N = prod(shape)
##    exp = reshape(M.exposure, N)
##    var = reshape(M.variance, N)
##    sig = reshape(M.comp, N)
##    if ndims(M.comp) == 1
##        Sigma = M.Sigma[1]
##    end
##    if ndims(M.comp) == 2
##        n, m = shape
##        Sigma = Array{Float64}(N,N)
##        for i in 1:n
##            for k in 1:n
##                for j in 1:m
##                    for l in 1:m
##                        Sigma[i+n*(j-1),k+n*(l-1)] = M.Sigma[1][i,k]*M.Sigma[2][j,l]
##                    end
##                end
##            end
##        end
##    end
##    D = diagm(1./exp./var) + Sigma
##    invD = inv(D)
##    v = invD*(sig./var)
##    result = v.^2./exp./exp./var
##    return reshape(result, shape)
##end
#
#function test1()
#    N = 10
#    m1 = zeros(N)
#    m1[1] = 1
#    m2 = rand(N)
#    m3 = rand(N)
#    E = 1:10
#    t1 = m1*E'
#    t2 = m2*E'
#    t3 = m3*E'
#    M = AdditiveModel([t1,t2,t3])
#    setpoissonbg!(M, 3e3, [.00001, 1., 2.])
#    I = fishermatrix(M)
#    Sigma = inv(I)
#    F = informationflux(M, 1)
#    println(F)
#    #println(1./inv(I)[1,1])
#    #println(sum(F)*3e3)
#end
#
#function test2()
#    nside = 16
#    npix = healpy.nside2npix(nside)
#    sig = collect(linspace(1, 2, npix))
#    i = healpy.ang2pix(nside, 0,0, lonlat=true)
#    sig[i+1] = 40
#    bkg = ones(npix)
#    Sigma = hpxcovariance(nside)*0.01 + opOnes(npix, npix)*0.01
#    M = SysMatrixModel(sig, bkg, Sigma)
#    for (i, expo) in enumerate(logspace(-3, 3, 20))
#        clf()
#        setexposure!(M, expo)
#        F = informationflux(M, solver="cg")
#        #healpy.mollview(F/mean(F), min = 0, max = 2)
#        q = quantile(F, WeightVec(F), 0.05)
#        mask = F .> q
#        healpy.mollview(mask, min = 0, max = 1)
#        savefig("test_$i.eps")
#    end
#    #imshow(F)
#    #show()
#    #println(1./inv(I)[1,1])
#    #println(sum(F)*3e3)
#end
#
#test2()
#quit()
#
#function binnedspectrum(spectrum::Function, edges::Array{Float64,1})
#    result = zeros(length(edges)-1)
#    for i in 1:length(edges)-1
#        result[i] = quadgk(spectrum, edges[i], edges[i+1])[1]
#    end
#    return result
#end
#
#function spectrummodel(spectra::Array{Function, 1}, edges::Array{Float64,1})
#    N = length(spectra)
#    mlist = []
#    for i in 1:N
#        template = binnedspectrum(spectra[i], edges)
#        mlist = vcat(mlist, TensorRank1(template))
#    end
#    return AdditiveModel(mlist)
#end
#
#
#function test1()
#    x = [1e-3,0., 1e-3]
#    y = [1.,]
#    m1 = TensorRank2(x, y)
#
#    x = [1.,1., 1.]
#    y = [1.,]
#    m2 = TensorRank2(x, y)
#
#    model = AdditiveModel([m1, m2])
#    setpoissonbg!(model, 1., [1., 1.])
#    setradiobg!(model, 23., 3e5, 3600., 15., 1.)
#
#    I = getfishermatrix(model)
#    Sigma = inv(I)
#    getupperlimitBL(Sigma, 1, 0)
#
#    N = 100
#    model = AdditiveModel([TensorRank1(rand(N)), TensorRank1(rand(N))])
#    setpoissonbg!(model, 1., [1.,1.])
#    Sigma = rand(N,N)
#    Sigma = Sigma + Sigma'
#    I = getmarginalfishermatrix(model, Sigma)
#
#    #edges = collect(logspace(-1, 1, 100))
#    #m = getspectrummodel([x->1./x, x->x], edges)
#end
#
## Keep information flux from point sources as independent outside component
#
#function test2()
#    halos = halogenerator()
#    signal!(halos)
#    big_halos = big(halos)
#    small_halos = small(halos)
#    dm_map = DMsignal(big_halos)
#    dm_spec = DMspectrum(model)
#    bg_map, bg_spec = bg()
#
#    TS = dm_map*dm_spec
#    TB = bg_map*bg_spec
#    Sigma = spatial covariance matrix
#    diffm = Model([TS, TB])
#    setbg!(diffm)
#    pointm = Podel(small_halos, diffm)
#    tS = frompoints(pointm)  # ∫ dΩ Ψ/(bg+Ψ)
#    tSS = frompoints(pointm)  # ∫ dΩ ΨΨ/(bg+Ψ)
#end
#
#
##test1()
