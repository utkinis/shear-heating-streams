using GLMakie
using Printf

@views avx(a) = 0.5 .* (a[1:end-1] .+ a[2:end])

@views function braeck_1D()
    # dimensionally independent physics
    h   = 1.0 # m
    σ0  = 1.0 # Pa
    τr  = 1.0 # s
    E_R = 1.0 # K
    # non-dimensional parameters
    npow    = 4
    h_L     = 5e-2
    T0_E_R  = 2e-2 
    Tbg_E_R = 5e-4
    σ0_σc   = 5e0
    τr_τd   = 1e-1
    # definitions
    μ0_μbg = exp(1 / T0_E_R - 1 / Tbg_E_R)
    Δp     = h_L + (1.0 - h_L) * μ0_μbg
    # dimensionally dependent physics
    L     = h / h_L
    σc    = σ0 / σ0_σc
    τd    = τr / τr_τd
    ttot  = min(0.6τr, 0.1τd)
    T0    = T0_E_R * E_R
    Tbg   = Tbg_E_R * E_R
    χ     = h^2 / τd
    A_C   = exp(E_R / T0) * σ0^(1 - npow) * T0^2 / (τr * σc^2 * E_R)
    AG    = exp(E_R / T0) * σ0^(1 - npow) / (2 * τr * Δp)
    Tmaxa = T0_E_R + σ0^2 * L / (2 * AG / A_C * h) / E_R
    @show Tmaxa
    # numerics
    nx     = 5000
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
    Vx    = zeros(nx + 1)
    # initialisation
    map!(T, xc) do x
        abs(x) <= h / 2 ? T0 : Tbg
    end
    σ = σ0
    # temporal evolution
    time_evo = Float64[0.0]
    Tmax_evo = Float64[maximum(T)/E_R]
    σ_evo    = Float64[1.0]
    Vmax_evo = Float64[0.0]
    sizehint!(time_evo, nt + 1)
    sizehint!(Tmax_evo, nt + 1)
    sizehint!(σ_evo, nt + 1)
    sizehint!(Vmax_evo, nt + 1)
    # figures
    fig = Figure(; resolution=(1000, 1400), fontsize=32)
    axs = (T        = Axis(fig[1, 1]; xlabel=L"x/h", ylabel=L"T/(E/R)", yscale=log10),
           σ_evo    = Axis(fig[2, 1]; xlabel=L"t/\tau_r", ylabel=L"\sigma/\sigma_0"),
           Tmax_evo = Axis(fig[3, 1]; xlabel=L"t/\tau_r", ylabel=L"T_\mathrm{max}/(E/R)", yscale=log10),
           Vmax_evo = Axis(fig[4, 1]; xlabel=L"t/\tau_r", ylabel=L"V_\mathrm{max}/(\sigma_0 h/\mu_0)", yscale=log10))
    plts = (T_ini    = lines!(axs.T, Point2.(xc ./ h, T ./ E_R); linewidth=4),
            T        = lines!(axs.T, Point2.(xc ./ h, T ./ E_R); linewidth=4),
            Tmaxa    = hlines!(axs.T, Tmaxa; linewidth=4, color=:gray, linestyle=:dash),
            σ_evo    = lines!(axs.σ_evo, Point2.(time_evo, σ ./ σ0); linewidth=4),
            Tmax_evo = lines!(axs.Tmax_evo, Point2.(time_evo, Tmax_evo ./ E_R); linewidth=4),
            Vmax_evo = lines!(axs.Vmax_evo, Point2.(time_evo, Vmax_evo); linewidth=4))
    limits!(axs.T, -2, 2, 0.8Tbg_E_R, 100Tmaxa)
    limits!(axs.σ_evo, 0.0, ttot / τr, 0.0, 1.0)
    limits!(axs.Tmax_evo, 0.0, ttot / τr, 0.8T0_E_R, 100Tmaxa)
    limits!(axs.Vmax_evo, 0.0, ttot / τr, 1e-5, 1.0e5)
    display(fig)
    # time loop
    it   = 1
    tcur = 0.0
    # iframe = 0
    while tcur < ttot
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
                dσ_dt = -AG / L * σ^(npow-1) * sum(exp.(-E_R ./ T)) * dx
                dt    = min(1e-3 * σ0 / abs(σ * dσ_dt), dt_diff, 1e-3 * τr)
                σ     = σ_old / (1 - dt * dσ_dt)
            end
            # temperature
            qTx[2:end-1] .= -χ * diff(T) ./ dx
            T .= T_old .+ dt * (.-diff(qTx) ./ dx .+ A_C .* σ^(npow + 1) .* exp.(-E_R ./ T))
        end
        # velocity
        cumsum!(Vx[2:end], (σ^npow .* exp.(-E_R ./ T) .+ (1 / AG) .* (σ - σ_old) / dt) .* dx)
        # evolution
        tcur += dt
        it   += 1
        push!(time_evo, tcur / τr)
        push!(Tmax_evo, maximum(T) / E_R)
        push!(Vmax_evo, maximum(Vx) / σ0^npow / h * exp(E_R / T0))
        push!(σ_evo, σ / σ0)
        # convergence check
        if any(.!isfinite.(T)) || !isfinite(σ)
            error("simulation failed")
        end
        # visualisation
        if it % nvis == 0
            # plots
            plts.T[1]        = Point2.(xc ./ h, T ./ E_R)
            plts.σ_evo[1]    = Point2.(time_evo, σ_evo)
            plts.Tmax_evo[1] = Point2.(time_evo, Tmax_evo)
            plts.Vmax_evo[1] = Point2.(time_evo, Vmax_evo)
            # save(@sprintf("anim/step_%04d.png", iframe), fig)
            # iframe += 1
            yield()
        end
    end
    @show (maximum(Tmax_evo) - T0_E_R) / (Tmaxa - T0_E_R)
    # run(`ffmpeg -framerate 15 -i anim/step_%04d.png -c libx264 -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2:color=white" -y adiabatic.mp4`)
    return
end

braeck_1D()
