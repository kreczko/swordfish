using Rockfish
using Base.Test
using StatsBase
using LinearOperators
using JLD

using PyCall
using PyPlot
@pyimport healpy
@pyimport healpy.fitsfunc as fitsfunc

function test1()
    nside = 32
    npix = healpy.nside2npix(nside)
    lon, lat = healpy.pix2ang(nside, 0:npix-1, lonlat=true)
    lon = mod(lon+180, 360)-180
    r = (lon.^2 + lat.^2).^0.5
    bkg = exp(-(lat./5).^2)*100+1
    sig = exp(-r/10)*10+1
    sig2 = exp(-r/10)
    #sig[30] = 3
    #sig[600] = 3

    flux = [sig, sig2]
    noise = bkg
    sigma = Rockfish.Σ_hpx(nside, sigma=10.)*0.01
    exposure = ones(npix)*1e3

    model = Rockfish.CoreModel(flux, noise, sigma, exposure)
    F, I = Rockfish.infoflux(model, solver="cg")
    F = Rockfish.effective(F, I, 1)

    #healpy.mollview(F)

    q = quantile(F, WeightVec(F), 0.05)
    mask = F .> q
    healpy.mollview(mask, min = 0, max = 1)

    savefig("test.eps")

    return true
end

function downsample(x::Vector{Float64}, nside::Int64)
    #nside_old = healpy.npix2nside(length(x))
    alm = healpy.map2alm(x)
    return healpy.alm2map(alm, nside)
end


function test2()
    nside = 64
    npix = healpy.nside2npix(nside)

    sig = fitsfunc.read_map("ADM.fits")*1e-20
    counts = fitsfunc.read_map("1GeV_healpix_counts.fits")
    exposure = fitsfunc.read_map("1GeV_healpix_exposure.fits")
    #sig = downsample(sig, nside)
    #counts = downsample(counts, nside)
    #exposure = downsample(exposure, nside)

    #lon, lat = healpy.pix2ang(nside, 0:npix-1, lonlat=true)
    #lon = mod(lon+180, 360)-180
    #r = (lon.^2 + lat.^2).^0.5

    flux = [sig]
    noise = counts./exposure
    exposure = ones(npix)*1e-30

    sigma = (
               Rockfish.Σ_hpx(nside, sigma=10.)*0.01 
             + Rockfish.Σ_hpx(nside, sigma=20.)*0.01
             + Rockfish.Σ_hpx(nside, sigma=30.)*0.01
             + Rockfish.Σ_hpx(nside, sigma=40.)*0.01
             + Rockfish.Σ_hpx(nside, sigma=50.)*0.01
             + Rockfish.Σ_hpx(nside, sigma=5.)*0.01
             + Rockfish.Σ_hpx(nside, sigma=2.)*0.01
             + Rockfish.Σ_hpx(nside, sigma=1.)*0.01
             + opOnes(npix, npix)*0.01
            )

    model = Rockfish.CoreModel(flux, noise, sigma, exposure)
    F, I = Rockfish.infoflux(model, solver="cg")
    F = Rockfish.effective(F, I, 1)

    clf()
    healpy.mollview(log10(F))
    savefig("F.eps")

    clf()
    q = quantile(F, WeightVec(F), 0.05)
    mask = F .> q
    healpy.mollview(mask, min = 0, max = 1)
    savefig("T.eps")

    return true
end

function test3()
    nside = 64
    npix = healpy.nside2npix(nside)

    sig = fitsfunc.read_map("ADM.fits")*1e-20
    counts = fitsfunc.read_map("1GeV_healpix_counts.fits")
    exposure = fitsfunc.read_map("1GeV_healpix_exposure.fits")

    flux = [sig/mean(sig)]
    noise = counts./exposure
    noise /= mean(noise)

    sigma = (
             #+ Rockfish.Σ_hpx(nside, sigma=01.)*0.01
             #+ Rockfish.Σ_hpx(nside, sigma=02.)*0.01
               Rockfish.Σ_hpx(nside, sigma=05., scale=noise)*0.01
             + Rockfish.Σ_hpx(nside, sigma=10., scale=noise)*0.01 
             + Rockfish.Σ_hpx(nside, sigma=20., scale=noise)*0.01
             + Rockfish.Σ_hpx(nside, sigma=30., scale=noise)*0.01
             + Rockfish.Σ_hpx(nside, sigma=40., scale=noise)*0.01
             + Rockfish.Σ_hpx(nside, sigma=50., scale=noise)*0.01
             #+ opOnes(npix, npix)*0.01
             + Rockfish.Σ_hpx(nside, sigma=Inf, scale=noise)*0.01
            )

    Ilist = Vector{Float64}()
    Flist = Vector{Vector{Float64}}()

    exp_list = collect(logspace(-4, 9, 14))

    for (i, exposure) in enumerate(exp_list)
        exposure = ones(npix)*exposure
        model = Rockfish.CoreModel(flux, noise, sigma, exposure)
        F, I = Rockfish.infoflux(model, solver="cg", maxiter=1000)
        F = Rockfish.effective(F, I, 1)
        I = Rockfish.effective(I, 1)
        push!(Ilist, I)
        push!(Flist, F)

        clf()
        healpy.mollview(log10(F))
        savefig("F_$i.eps")

        clf()
        q = quantile(F, WeightVec(F), 0.05)
        mask = F .> q
        healpy.mollview(mask, min = 0, max = 1)
        savefig("T_$i.eps")
    end

    save("I.jld", "Ilist", Ilist, "Flist", Flist)

    return true
end

# write your own tests here
@test test3()



####################################
####################################
### NOTE: This needs some clean-up
####################################
####################################


#######
# Sampling Tests
#######

#using PyPlot
#s = FullSphere()
#a = sample(s, 100000)
#using PyPlot
#PyPlot.plt[:hist](a, bins=100, log=true)
#show()
#quit()

#@time s = Inversion2D((x,y) -> x.^-2.5 + 0.*y, logspace(-1, 2, 100), logspace(-10, 1, 50), logint = (true, false))
#@time x, y = sample(s, 100000)
#PyPlot.plt[:hist](log10(x), bins=100, log=true)

#@time s = Inversion1D(x -> x.^-2.5, logspace(-1, 2, 10), logint = true)
#@time x = sample(s, 100000)
#PyPlot.plt[:hist](log10(x), bins=100, log=true)
#show()


# int x x' sum_lm Ylm(x) Ylm(x') phi(x) phi(x')
# int x x' sum_l Pl(x*x') phi(x) phi(x')

#using JLD
#@save "out.jld" lgrid fgrid xmin
#quit()

#i = 8
#l = lgrid[i]
#println(l)
#x = collect(linspace(xmin[i], 1, 1000))
#plot(x, fgrid[i](x))
#int = trapz(x, fgrid[i](x))
#println(int)
#int = trapz(x, fgrid[i](x).^2)
#println(int)
#show()
#quit()

#m = SkyMapHealpix(512)
#patch = getpatch(m, halo, emission)
#apply!(m, patch)
#healpy.mollview(log(m.val))
#alm = healpy.map2alm(m.val)
#cl = healpy.alm2cl(alm)
#l = collect(1:length(cl)-1)
#loglog(l, cl[2:end].*l.^2)

#import PyPlot: loglog, show

#loglog(emission.psi, emission.intensity)
#loglog(emission2.psi, emission2.intensity)
#loglog(prof.r, prof.ρ)
#loglog(prof2.r, prof2.ρ)
#show()
#quit()

#function test()
#    loc = getlocation(lat = 0., lon = 0., D = 800.5)
#
#    prof = getprofile(1000)
#    halo = SphHalo(loc, prof)
#    func = h->h.ρ.^2
#    emission = getemission(halo, func)
#
#    l = [Int(floor(l0)) for l0 in logspace(0, 5, 20)]
#    #@time cl2b = [getClb(emission, l0) for l0 in l]
#    #@time cl2b = [getClb(emission, l0) for l0 in l]
#    #@time cl2b = [getClb(emission, l0) for l0 in l]
#    #@time cl2b = [getClb(emission, l0) for l0 in l]
#    #loglog(l, pi*cl2b.*l.^2)
#
#    @time l, y = getClc(emission)
#    @time l, y = getClc(emission)
#    @time l, y = getClc(emission)
#    loglog(l, pi*y.*l.^2)
#
#    show()
#end
