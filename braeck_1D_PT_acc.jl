using GLMakie
using Printf
using LazyArrays

@views avx(a) = 0.5 .* (a[1:end-1] .+ a[2:end])

@views function braeck_1D()
    # dimensionally independent physics
    h   = 1.0 # m
    σ0  = 1.0 # Pa
    τr  = 1.0 # s
    E_R = 1.0 # K
    # non-dimensional parameters
    npow    = 3
    h_L     = 5e-2
    T0_E_R  = 2e-2
    Tbg_E_R = 5e-4
    σ0_σc   = 5e0
    τr_τd   = 1e-3
    G_σ0    = 1.0
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
    G     = G_σ0 * σ0
    A     = exp(E_R / T0) * σ0^(1 - npow) / (2 * G * Δp * τr)
    C     = (σc / T0)^2 / (2 * G * Δp) * E_R
    Tmaxa = T0_E_R + σ0^2 * L / (2 * G * C * h) / E_R
    @show Tmaxa
    # numerics
    nx     = 2000
    niter  = 20nx
    nvis   = 10
    ncheck = ceil(Int, 1nx)
    ϵtol   = 1e-8
    # preprocessing
    dx      = L / nx
    xv      = LinRange(-L / 2, L / 2, nx + 1)
    xc      = avx(xv)
    dt_diff = dx^2 / χ / 20.1
    dt      = dt_diff
    nt      = ceil(Int, ttot / dt)
    # PT params
    r = 0.9
    lτ_re_mech = 0.15L / π
    vdτ = dx / sqrt(3.1)
    θ_dτ = lτ_re_mech * (r + 4 / 3) / vdτ
    nudτ = vdτ * lτ_re_mech
    dτ_r = 1.0 / (θ_dτ + 1.0)
    # fields
    # thermo
    T     = zeros(nx + 2)
    T_old = zeros(nx + 2)
    qTx   = zeros(nx + 1)
    # mechanics
    σxy     = zeros(nx + 1)
    σxy_old = zeros(nx + 1)
    Vy      = zeros(nx + 2)
    μ       = zeros(nx + 1)
    # initialisation
    map!(T[2:end-1], xc) do x
        abs(x) <= h / 2 ? T0 : Tbg
    end
    T[[1, end]] .= T[[2, end - 1]]
    σxy .= σ0
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
           Vy       = Axis(fig[2, 1]; xlabel=L"x/h", ylabel=L"v_y"),
           σ_evo    = Axis(fig[3, 1]; xlabel=L"t/\tau_r", ylabel=L"\sigma/\sigma_0"),
           Tmax_evo = Axis(fig[4, 1]; xlabel=L"t/\tau_r", ylabel=L"T_\mathrm{max}/(E/R)", yscale=log10),
           Vmax_evo = Axis(fig[5, 1]; xlabel=L"t/\tau_r", ylabel=L"V_\mathrm{max}/(\sigma_0 h/\mu_0)", yscale=log10))
    plts = (T_ini    = lines!(axs.T, Point2.(xc ./ h, T[2:end-1] ./ E_R); linewidth=4),
            T        = lines!(axs.T, Point2.(xc ./ h, T[2:end-1] ./ E_R); linewidth=4),
            Tmaxa    = hlines!(axs.T, Tmaxa; linewidth=4, color=:gray, linestyle=:dash),
            Vy       = lines!(axs.Vy, Point2.(xc ./ h, Vy[2:end-1]); linewidth=4),
            σ_evo    = lines!(axs.σ_evo, Point2.(time_evo, maximum(σxy) / σ0); linewidth=4),
            Tmax_evo = lines!(axs.Tmax_evo, Point2.(time_evo, Tmax_evo ./ E_R); linewidth=4),
            Vmax_evo = lines!(axs.Vmax_evo, Point2.(time_evo, Vmax_evo); linewidth=4))
    limits!(axs.T, -2, 2, 0.5Tbg_E_R, 2Tmaxa)
    limits!(axs.Vy, -2, 2, -5e3, 5e3)
    limits!(axs.σ_evo, 0.0, ttot / τr, 0.0, 1.0)
    limits!(axs.Tmax_evo, 0.0, ttot / τr, 0.5Tbg_E_R, 2Tmaxa)
    limits!(axs.Vmax_evo, 0.0, ttot / τr, 1e-5, 1.0e5)
    display(fig)
    # time loop
    it   = 1
    tcur = 0.0
    # iframe = 0
    while tcur < ttot
        println("it = $it")
        σxy_old .= σxy
        T_old .= T
        # time step
        dσ_dt = maximum(abs.(-(A * G) .* σxy .^ npow .* exp.(-E_R ./ avx(T)) .+ G .* diff(Vy) ./ dx))
        dt    = min(1e-3 * σ0 / dσ_dt, dt_diff, 1e-3 * τr)
        # iteration loop
        for iter in 1:niter
            # stress
            μ   .= (1 / A) .* exp.(E_R ./ avx(T)) .* σxy .^ (1 - npow)
            σxy .+= (.-(σxy .- σxy_old) ./ (G * dt) .- σxy ./ μ .+ Diff(Vy) ./ dx) ./ (1 / (G * dt) .+ 1 ./ μ) .* dτ_r
            # velocity
            Vy[2:end-1] .+= (Diff(σxy) ./ dx) .* (1 / (G * dt) .+ 1 ./ max.(μ[1:end-1], μ[2:end])) .* nudτ
            Vy[[1, end]] .= .-Vy[[2, end - 1]]
            # temperature
            qTx[2:end-1] .= -χ .* Diff(T[2:end-1]) ./ dx
            T[2:end-1]   .= T_old[2:end-1] .+ dt .* (.-Diff(qTx) ./ dx .+ (1 / C) .* avx(σxy .^ 2 ./ μ))
            T[[1, end]]  .= T[[2, end - 1]]
            if iter % ncheck == 0
                err = maximum(abs.(Diff(σxy) ./ dx)) / σ0 * h
                @printf("  iter / nx = %.1f, err = %1.3e\n", iter / nx, err)
                if err < ϵtol
                    break
                end
            end
        end
        # other velocity
        # cumsum!(Vx[2:end], (σxy.^npow .* exp.(-E_R ./ T) .+ (1 / A / G) .* (σxy .- σxy_old) / dt) .* dx)
        tcur += dt
        it   += 1
        # evolution
        push!(time_evo, tcur / τr)
        push!(Tmax_evo, maximum(T) / E_R)
        push!(Vmax_evo, maximum(Vy) / (σ0^npow * h * A * exp(-E_R / T0)))
        push!(σ_evo, maximum(σxy) / σ0)
        # convergence check
        if any(.!isfinite.(T)) || any(.!isfinite.(σxy))
            error("simulation failed")
        end
        # visualisation
        if it % nvis == 0
            # plots
            plts.T[1]        = Point2.(xc ./ h, T[2:end-1] ./ E_R)
            plts.Vy[1]       = Point2.(xc ./ h, Vy[2:end-1])
            plts.σ_evo[1]    = Point2.(time_evo, σ_evo)
            plts.Tmax_evo[1] = Point2.(time_evo, Tmax_evo)
            plts.Vmax_evo[1] = Point2.(time_evo, Vmax_evo)
            # autolimits!(axs.T)
            # autolimits!(axs.Vy)
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

braeck_1D()
