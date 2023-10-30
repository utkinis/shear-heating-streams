using GLMakie
using Printf

@views av1(a) = 0.5 .* (a[1:end-1] .+ a[2:end])
@views avx(a) = 0.5 .* (a[1:end-1, :] .+ a[2:end, :])
@views avy(a) = 0.5 .* (a[:, 1:end-1] .+ a[:, 2:end])
@view av4(a) = 0.25 .* (a[1:end-1, 1:end-1] .+ a[2:end, 1:end-1] .+ a[2:end, 2:end] .+ a[1:end-1, 2:end])

@views function braeck_2D()
    # dimensionally independent physics
    h   = 1.0 # m
    σ0  = 1.0 # Pa
    τr  = 1.0 # s
    E_R = 1.0 # K
    # non-dimensional parameters
    npow    = 1
    h_Lx    = 5e-2
    h_Ly    = 5e-2
    T0_E_R  = 2e-2
    Tbg_E_R = 5e-4
    σ0_σc   = 5e0
    τr_τd   = 1e-3
    # definitions
    μ0_μbg = exp(1 / T0_E_R - 1 / Tbg_E_R)
    Δp     = h_Lx + (1.0 - h_Lx) * μ0_μbg
    # dimensionally dependent physics
    Lx    = h / h_Lx
    Ly    = h / h_Ly
    σc    = σ0 / σ0_σc
    τd    = τr / τr_τd
    ttot  = min(0.6τr, 0.1τd)
    T0    = T0_E_R * E_R
    Tbg   = Tbg_E_R * E_R
    χ     = h^2 / τd
    A_C   = exp(E_R / T0) * σ0^(1 - npow) * T0^2 / (τr * σc^2 * E_R)
    AG    = exp(E_R / T0) * σ0^(1 - npow) / (2 * τr * Δp)
    Tmaxa = T0_E_R + σ0^2 * Lx / (2 * AG / A_C * h) / E_R
    # numerics
    nx    = 250
    ny    = ceil(Int, nx * Ly / Lx)
    niter = 5
    nvis  = 10
    ϵtol  = 1e-12
    # preprocessing
    dx      = Lx / nx
    dy      = Ly / ny
    xv      = LinRange(-Lx / 2, Lx / 2, nx + 1)
    yv      = LinRange(-Ly / 2, Ly / 2, ny + 1)
    xc      = av1(xv)
    yc      = av1(yv)
    dt_diff = min(dx, dy)^2 / χ / 20.1
    dt      = dt_diff
    nt      = ceil(Int, ttot / dt)
    # fields
    # thermo
    T     = zeros(nx, ny)
    T_old = zeros(nx, ny)
    qTx   = zeros(nx + 1, ny)
    qTy   = zeros(nx, ny + 1)
    # mechanics
    Pr = zeros(nx, ny)
    τxx = zeros(nx, ny)
    τyy = zeros(nx, ny)
    τxy = zeros(nx + 1, ny + 1)
    τxx_old = copy(τxx)
    τyy_old = copy(τyy)
    τxy_old = copy(τxy)
    Vx = zeros(nx + 1, ny)
    Vy = zeros(nx, ny + 1)
    ∇V = zeros(nx, ny)
    τII = zeros(nx, ny)
    # initialisation
    broadcast!(T, xc, yc') do x, _
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
    fig = Figure(; resolution=(1000, 1800), fontsize=32)
    gl = fig[1, 1] = GridLayout()
    axs = (T        = Axis(gl[1, 1][1, 1]; xlabel=L"x/h", ylabel="y/h", title=L"T/(E/R)", aspect=DataAspect()),
           T_sl     = Axis(gl[2, 1]; xlabel=L"t/\tau_r", ylabel=L"T/(E/R)"),
           σ_evo    = Axis(gl[3, 1]; xlabel=L"t/\tau_r", ylabel=L"\sigma/\sigma_0"),
           Tmax_evo = Axis(gl[4, 1]; xlabel=L"t/\tau_r", ylabel=L"T_\mathrm{max}/(E/R)", yscale=log10),
           Vmax_evo = Axis(gl[5, 1]; xlabel=L"t/\tau_r", ylabel=L"V_\mathrm{max}/(\sigma_0 h/\mu_0)", yscale=log10))
    plts = (T        = heatmap!(axs.T, xc ./ h, yc ./ h, T ./ E_R; colormap=:turbo),
            T_sl_ini = lines!(axs.T_sl, Point2.(xc ./ h, T[:, ny÷2] ./ E_R); linewidth=4),
            T_sl     = lines!(axs.T_sl, Point2.(xc ./ h, T[:, ny÷2] ./ E_R); linewidth=4),
            Tmaxa    = hlines!(axs.T_sl, Tmaxa; linewidth=4, color=:gray, linestyle=:dash),
            σ_evo    = lines!(axs.σ_evo, Point2.(time_evo, σ ./ σ0); linewidth=4),
            Tmax_evo = lines!(axs.Tmax_evo, Point2.(time_evo, Tmax_evo ./ E_R); linewidth=4),
            Vmax_evo = lines!(axs.Vmax_evo, Point2.(time_evo, Vmax_evo); linewidth=4))
    rowsize!(gl, 1, Relative(0.6))
    limits!(axs.T_sl, -2, 2, 0.5Tbg_E_R, 2Tmaxa)
    limits!(axs.σ_evo, 0.0, ttot / τr, 0.0, 1.0)
    limits!(axs.Tmax_evo, 0.0, ttot / τr, 0.5Tbg_E_R, 2Tmaxa)
    limits!(axs.Vmax_evo, 0.0, ttot / τr, 1e-5, 1.0e5)
    Colorbar(gl[1, 1][1, 2], plts.T)
    display(fig)
    # time loop
    it   = 1
    tcur = 0.0
    # iframe = 0
    while tcur < ttot
        τxx_old .= τxx
        τyy_old .= τyy
        τxy_old .= τxy
        σ_old = σ
        T_old .= T
        for iter in 1:niter
            # stress
            dσ_dt = -AG / Lx * σ^(npow - 1) * sum(exp.(-E_R ./ T[:, ny÷2])) * dx
            dt    = min(1e-3 * σ0 / abs(σ * dσ_dt), dt_diff, 1e-3 * τr)
            σ     = σ_old / (1 - dt * dσ_dt)
            # stress 2
            τII .= sqrt.(0.5 .* (τxx .^ 2 .+ τyy .^ 2) .+ av4(τxy) .^ 2)
            ∇V .= diff(Vx; dims=1) ./ dx + diff(Vy; dims=2) ./ dy
            μ = (1 / A) .* exp.(E_R ./ T) .* τII .^ (1 - npow)
            τxx .* (1 ./ dt .+ G ./ μ) .= τxx_old ./ dt .+ 2.0 .* G .* (diff(Vx; dims=1) ./ dx .- ∇V ./ 3.0)
            # temperature
            qTx[2:end-1, :] .= -χ * diff(T; dims=1) ./ dx
            qTy[:, 2:end-1] .= -χ * diff(T; dims=2) ./ dy
            T .= T_old .+ dt * (.-diff(qTx; dims=1) ./ dx .- diff(qTy; dims=2) ./ dy .+ A_C .* σ .^ (npow + 1) .* exp.(-E_R ./ T))
        end
        # velocity
        # cumsum!(Vx[2:end], (σ^npow .* exp.(-E_R ./ T) .+ (1 / AG) .* (σ - σ_old) / dt) .* dx)
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
            plts.T[3]        = T ./ E_R
            plts.T_sl[1]     = Point2.(xc ./ h, T[:, ny÷2] ./ E_R)
            plts.σ_evo[1]    = Point2.(time_evo, σ_evo)
            plts.Tmax_evo[1] = Point2.(time_evo, Tmax_evo)
            plts.Vmax_evo[1] = Point2.(time_evo, Vmax_evo)
            # autolimits!(axs.T)
            # autolimits!(axs.σ_evo)
            # autolimits!(axs.Tmax_evo)
            # autolimits!(axs.Vmax_evo)
            # save(@sprintf("anim/step_%04d.png", iframe), fig)
            # iframe += 1
            yield()
        end
    end
    @show (maximum(Tmax_evo) - T0_E_R) / (Tmaxa - T0_E_R)
    # run(`ffmpeg -framerate 15 -i anim/step_%04d.png -c libx264 -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2:color=white" -y adiabatic.mp4`)
    return
end

braeck_2D()
