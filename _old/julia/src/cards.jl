#!/usr/bin/env julia

# Rockfish

type PoissonExperiment
    angres::Function
    aeff::Function
    instrbkg::Function
    erange::Tuple{Float64,Float64}  # GeV
end

function get(fluxes::Vector{Vector{Float64}}, bkg::Vector{Float64}, Σ::Matrix{Float64}, experiment::PoissonExperiment)
    return model()
end
