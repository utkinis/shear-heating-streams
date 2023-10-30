using GLMakie
using Printf

@views avx(a) = 0.5 .* (a[1:end-1] .+ a[2:end])

@views function braeck_1D(params)
    # dimensionally independent physics
    h   = 1.0 # m
    σ0  = 1.0 # Pa
    τr  = 1.0 # s
    E_R = 1.0 # K
    # non-dimensional parameters
    npow             = 1
    h_L              = 2e-2
    T0_E_R           = 2e-2
    Tbg_E_R          = 5e-4
    (; σ0_σc, τr_τd) = params
    # definitions
    μ0_μbg = exp(1 / T0_E_R - 1 / Tbg_E_R)
    Δp     = h_L + (1.0 - h_L) * μ0_μbg
    # dimensionally dependent physics
    L    = h / h_L
    σc   = σ0 / σ0_σc
    τd   = τr / τr_τd
    ttot = min(0.6τr, 0.1τd)
    T0   = T0_E_R * E_R
    Tbg  = Tbg_E_R * E_R
    χ    = h^2 / τd
    A_C  = exp(E_R / T0) * σ0^(1 - npow) * T0^2 / (τr * σc^2 * E_R)
    AG   = exp(E_R / T0) * σ0^(1 - npow) / (2 * τr * Δp)
    # numerics
    nx     = 1000
    niter  = 5
    nvis   = 100
    ϵtol   = 1e-12
    method = :varchange
    # preprocessing
    dx      = L / nx
    xv      = LinRange(-L / 2, L / 2, nx + 1)
    xc      = avx(xv)
    dt_diff = dx^2 / χ / 10.1
    dt      = dt_diff
    nt      = ceil(Int, ttot / dt)
    # fields
    T     = zeros(nx)
    T_old = zeros(nx)
    qTx   = zeros(nx + 1)
    # initialisation
    σ = σ0
    map!(T, xc) do x
        abs(x) <= h / 2 ? T0 : Tbg
    end
    time_evo = Float64[0.0]
    Tmax_evo = Float64[maximum(T)/E_R]
    sizehint!(time_evo, nt + 1)
    sizehint!(Tmax_evo, nt + 1)
    # time loop
    it       = 1
    tcur     = 0.0
    sim_done = false
    # iframe = 0
    while !sim_done
        σ_old = σ
        T_old .= T
        for iter in 1:niter
            # stress
            if method == :varchange
                if npow > 1
                    Ψ_old = σ_old^(1 - npow) / (1 - npow)
                    dΨ_dt = -AG / L * sum(exp.(-E_R ./ T)) * dx
                    dt    = min(1e-3 * σ0 / abs(σ^npow * dΨ_dt), dt_diff, 1e-3 * τr)
                    Ψ     = Ψ_old + dt * dΨ_dt
                    σ     = (Ψ * (1 - npow))^(1 / (1 - npow))
                else
                    Ψ_old = log(σ_old)
                    dΨ_dt = -AG / L * sum(exp.(-E_R ./ T)) * dx
                    dt    = min(1e-3 * σ0 / abs(σ^npow * dΨ_dt), dt_diff, 1e-3 * τr)
                    Ψ     = Ψ_old + dt * dΨ_dt
                    σ     = exp(Ψ)
                end
            else
                dσ_dt = -AG / L * σ^(npow - 1) * sum(exp.(-E_R ./ T)) * dx
                dt    = min(1e-3 * σ0 / abs(σ * dσ_dt), dt_diff, 1e-3 * τr)
                σ     = σ_old / (1 - dt * dσ_dt)
            end
            # temperature
            qTx[2:end-1] .= -χ * diff(T) ./ dx
            T .= T_old .+ dt * (.-diff(qTx) ./ dx .+ A_C .* σ^(npow + 1) .* exp.(-E_R ./ T))
        end
        # evolution
        tcur += dt
        it   += 1
        push!(time_evo, tcur / τr)
        push!(Tmax_evo, maximum(T) / E_R)
        # convergence check
        if any(.!isfinite.(T)) || !isfinite(σ)
            error("simulation failed at it = $it")
        end
        # stop if no localisation
        if Tmax_evo[end] < 0.8maximum(Tmax_evo)
            sim_done = true
        end
        # stop if time passed
        if tcur > ttot
            sim_done = true
        end
    end
    ΔTmax  = maximum(Tmax_evo) - T0_E_R
    ΔTmaxa = σ0^2 * L / (2 * AG / A_C * h) / E_R
    return ΔTmax, ΔTmaxa
end

σ0_σc_rng = 10 .^ LinRange(-4, 4, 50)
τr_τd_rng = 10 .^ LinRange(-5, 3, 50)

ΔTmax  = zeros(length(σ0_σc_rng), length(τr_τd_rng))
ΔTmaxa = zeros(length(σ0_σc_rng), length(τr_τd_rng))

for (iτ, τr_τd) in enumerate(τr_τd_rng)
    for (iσ, σ0_σc) in enumerate(σ0_σc_rng)
        @printf(" iσ = %d/%d, iτ = %d/%d, [τr_τd = %1.3e, σ0_σc = %1.3e]\n", iσ, length(σ0_σc_rng), iτ, length(τr_τd_rng), τr_τd, σ0_σc)
        ΔTmax[iσ, iτ], ΔTmaxa[iσ, iτ] = braeck_1D((; σ0_σc, τr_τd))
    end
end

fig = Figure(; resolution=(1000, 1000), fontsize=32)
ax  = Axis(fig[1, 1]; xlabel=L"\sigma/\sigma_0", ylabel=L"\tau_r/\tau_d", xscale=log10, yscale=log10)
limits!(ax, σ0_σc_rng[1], σ0_σc_rng[end], τr_τd_rng[1], τr_τd_rng[end])
cf  = contourf!(ax, σ0_σc_rng, τr_τd_rng, log10.(ΔTmax ./ ΔTmaxa); colormap=:turbo, levels=-7:0.5:1)
cc  = contour!(ax, σ0_σc_rng, τr_τd_rng, log10.(ΔTmax ./ ΔTmaxa); color=:black, linewidth=2, levels=-7:0.5:1)
Colorbar(fig[1, 2], cf; label=L"log_{10}(\Delta T_\mathrm{max}/\Delta T^a_\mathrm{max})")

save("phase_portrait.png", fig)
