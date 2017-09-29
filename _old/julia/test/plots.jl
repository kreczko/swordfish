#!/usr/bin/env julia

using JLD
using StatsBase
using PyPlot
using PyCall
@pyimport healpy


data = load("I.jld")
Ilist = data["Ilist"]
Flist = data["Flist"]

N = length(Ilist)

semilogy(1:N, Ilist)
xlabel("Exposure")
ylabel("Eff. information")
savefig("out.pdf")

for i in 1:N
    F = Flist[i]
    println(i)
    figure(figsize=(4,5.5))
    healpy.mollview(log10(F), sub=211, title="Effective information flux")
    q1 = quantile(F, WeightVec(F), 0.003)
    q2 = quantile(F, WeightVec(F), 0.05)
    q3 = quantile(F, WeightVec(F), 0.32)
    q = Int.(F.>q1)+Int.(F.>q2)+Int.(F.>q3)
    healpy.mollview(q, min = 0, max = 3, sub=212, title="Quantiles", cbar=false)
    outfile = @sprintf("out_%02d.pdf",i)
    savefig(outfile)
end
