using GLMakie
using Printf
using LazyArrays

@views av1(a) = 0.5 .* (a[1:end-1] .+ a[2:end])
@views avx(a) = 0.5 .* (a[1:end-1, :] .+ a[2:end, :])
@views avy(a) = 0.5 .* (a[:, 1:end-1] .+ a[:, 2:end])
@views av4(a) = 0.25 .* (a[1:end-1, 1:end-1] .+ a[2:end, 1:end-1] .+ a[2:end, 2:end] .+ a[1:end-1, 2:end])

@views function braeck_2D()
    # dimensionally independent physics
    h   = 1.0 # m
    σ0  = 1.0 # Pa
    τr  = 1.0 # s
    E_R = 1.0 # K
    # non-dimensional parameters
    npow    = 3
    h_Lx    = 5e-2
    h_Ly    = 5e-2
    T0_E_R  = 2e-2
    Tbg_E_R = 5e-4
    σ0_σc   = 5e0
    τr_τd   = 1e-3
    G_σ0    = 1.0
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
    G     = G_σ0 * σ0
    A     = exp(E_R / T0) * σ0^(1 - npow) / (2 * G * Δp * τr)
    C     = (σc / T0)^2 / (2 * G * Δp) * E_R
    Tmaxa = T0_E_R + σ0^2 * Lx / (2 * G * C * h) / E_R
    @show Tmaxa
    # numerics
    nx     = 200
    ny     = 200
    niter  = 20min(nx, ny)
    nvis   = 1
    ncheck = ceil(Int, 1min(nx, ny))
    ϵtol   = 1e-4
    # preprocessing
    dx      = Lx / nx
    dy      = Ly / ny
    xv      = LinRange(-Lx / 2, Lx / 2, nx + 1)
    yv      = LinRange(-Ly / 2, Ly / 2, ny + 1)
    xc      = av1(xv)
    yc      = av1(yv)
    dt_diff = min(dx, dy)^2 / χ / 10.1
    dt      = dt_diff
    nt      = ceil(Int, ttot / dt)
    # PT params
    r          = 0.9
    lτ_re_mech = 0.15min(Lx, Ly) / π
    vdτ        = min(dx, dy) / sqrt(3.1)
    θ_dτ       = lτ_re_mech * (r + 4 / 3) / vdτ
    nudτ       = vdτ * lτ_re_mech
    dτ_r       = 1.0 / (θ_dτ + 1.0)
    # fields
    # thermo
    T     = zeros(nx + 2, ny + 2)
    T_old = zeros(nx + 2, ny + 2)
    qTx   = zeros(nx + 1, ny)
    qTy   = zeros(nx, ny + 1)
    # mechanics
    Pr      = zeros(nx, ny)
    τxx     = zeros(nx, ny)
    τyy     = zeros(nx, ny)
    τxy     = zeros(nx + 1, ny + 1)
    τII     = zeros(nx + 1, ny + 1)
    τxx_old = zeros(nx, ny)
    τyy_old = zeros(nx, ny)
    τxy_old = zeros(nx + 1, ny + 1)
    Vx      = zeros(nx + 1, ny + 2)
    Vy      = zeros(nx + 2, ny + 1)
    ∇V      = zeros(nx, ny)
    μ       = zeros(nx + 1, ny + 1)
    # initialisation
    broadcast!(T[2:end-1, 2:end-1], xc, yc') do x, y
        abs(x) <= h / 2 ? T0 : Tbg
    end
    T[[1, end], :] .= T[[2, end - 1], :]
    T[:, [1, end]] .= T[:, [2, end - 1]]
    τxy .= σ0
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
    fig = Figure(; resolution=(1000, 1600), fontsize=32)
    gl  = fig[1, 1] = GridLayout()
    rowsize!(gl, 1, Relative(0.4))
    axs = (T        = Axis(gl[1, 1][1, 1][1, 1]; xlabel=L"x/h", ylabel=L"y/h", title=L"T/(E/R)", aspect=DataAspect()),
           T_sl     = Axis(gl[1, 1][1, 2][1, 1]; xlabel=L"x/h", ylabel=L"T/(E/R)", yscale=log10),
           Vy_sl    = Axis(gl[1, 1][1, 2][2, 1]; xlabel=L"x/h", ylabel=L"v_y"),
           σ_evo    = Axis(gl[2, 1]; xlabel=L"t/\tau_r", ylabel=L"\sigma/\sigma_0"),
           Tmax_evo = Axis(gl[3, 1]; xlabel=L"t/\tau_r", ylabel=L"T_\mathrm{max}/(E/R)", yscale=log10),
           Vmax_evo = Axis(gl[4, 1]; xlabel=L"t/\tau_r", ylabel=L"V_\mathrm{max}/(\sigma_0 h/\mu_0)", yscale=log10))
    plts = (T        = heatmap!(axs.T, xc, yc, T[2:end-1, 2:end-1] ./ E_R; colormap=:turbo),
            T_ini_sl = lines!(axs.T_sl, Point2.(xc ./ h, T[2:end-1, ny÷2] ./ E_R); linewidth=4),
            T_sl     = lines!(axs.T_sl, Point2.(xc ./ h, T[2:end-1, ny÷2] ./ E_R); linewidth=4),
            Tmaxa    = hlines!(axs.T_sl, Tmaxa; linewidth=4, color=:gray, linestyle=:dash),
            Vy_sl    = lines!(axs.Vy_sl, Point2.(xc ./ h, Vy[2:end-1, ny÷2]); linewidth=4),
            σ_evo    = lines!(axs.σ_evo, Point2.(time_evo, maximum(τxy) / σ0); linewidth=4),
            Tmax_evo = lines!(axs.Tmax_evo, Point2.(time_evo, Tmax_evo ./ E_R); linewidth=4),
            Vmax_evo = lines!(axs.Vmax_evo, Point2.(time_evo, Vmax_evo); linewidth=4))
    limits!(axs.T_sl, -2, 2, 0.5Tbg_E_R, 2Tmaxa)
    limits!(axs.Vy_sl, -2, 2, -5e3, 5e3)
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
        τxx_old .= τxx
        τyy_old .= τyy
        τxy_old .= τxy
        T_old .= T
        # time step
        τII[2:end-1, 2:end-1] .= sqrt.(0.5 .* (av4(τxx) .^ 2 .+ av4(τyy) .^ 2) .+ τxy[2:end-1, 2:end-1] .^ 2)
        τII[[1, end], :] .= τII[[2, end - 1], :]
        τII[:, [1, end]] .= τII[:, [2, end - 1]]
        dσ_dt = maximum(abs.(-(A * G) .* τII .^ npow .* exp.(-E_R ./ av4(T)) .+ G .* (Diff(Vx; dims=2) ./ dy .+ Diff(Vy; dims=1) ./ dx)))
        dt = min(1e-4 * σ0 / dσ_dt, dt_diff, 1e-3 * τr)
        # iteration loop
        for iter in 1:niter
            # stress
            τII[2:end-1, 2:end-1] .= sqrt.(0.5 .* (av4(τxx) .^ 2 .+ av4(τyy) .^ 2) .+ τxy[2:end-1, 2:end-1] .^ 2)
            τII[[1, end], :] .= τII[[2, end - 1], :]
            τII[:, [1, end]] .= τII[:, [2, end - 1]]
            μ .= (1 / A) .* exp.(E_R ./ av4(T)) .* τII .^ (1 - npow)
            ∇V .= Diff(Vx[:, 2:end-1]; dims=1) ./ dx .+ Diff(Vy[2:end-1, :]; dims=2) ./ dy
            Pr .-= ∇V ./ (1 / (G * dt) .+ 1 ./ av4(μ)) .* (r / θ_dτ)
            τxx .+= (.-(τxx .- τxx_old) ./ (G * dt) .- τxx ./ av4(μ) .+ 2.0 .* (Diff(Vx[:, 2:end-1]; dims=1) ./ dx .- ∇V ./ 3.0)) ./ (1 / (G * dt) .+ 1 ./ av4(μ)) .* dτ_r
            τyy .+= (.-(τyy .- τyy_old) ./ (G * dt) .- τyy ./ av4(μ) .+ 2.0 .* (Diff(Vy[2:end-1, :]; dims=2) ./ dy .- ∇V ./ 3.0)) ./ (1 / (G * dt) .+ 1 ./ av4(μ)) .* dτ_r
            τxy .+= (.-(τxy .- τxy_old) ./ (G * dt) .- τxy ./ μ .+ Diff(Vx; dims=2) ./ dy .+ Diff(Vy; dims=1) ./ dx) ./ (1 / (G * dt) .+ 1 ./ μ) .* dτ_r
            # velocity
            Vx[2:end-1, 2:end-1] .+= (.-Diff(Pr; dims=1) ./ dx .+ Diff(τxx; dims=1) ./ dx .+ Diff(τxy[2:end-1, :]; dims=2) ./ dy) .*
                                     (1 / (G * dt) .+ 1 ./ max.(μ[1:end-2, 1:end-1], μ[2:end-1, 1:end-1], μ[3:end, 1:end-1],
                                                                μ[1:end-2, 2:end], μ[2:end-1, 2:end], μ[3:end, 2:end])) .* nudτ
            Vy[2:end-1, 2:end-1] .+= (.-Diff(Pr; dims=2) ./ dy .+ Diff(τyy; dims=2) ./ dy .+ Diff(τxy[:, 2:end-1]; dims=1) ./ dx) .*
                                     (1 / (G * dt) .+ 1 ./ max.(μ[1:end-1, 1:end-2], μ[1:end-1, 2:end-1], μ[1:end-1, 3:end],
                                                                μ[2:end, 1:end-2], μ[2:end, 2:end-1], μ[2:end, 3:end])) .* nudτ
            Vx[:, [1, end]] .= .-Vx[:, [2, end - 1]]
            Vy[[1, end], :] .= .-Vy[[2, end - 1], :]
            # temperature
            qTx[2:end-1, :] .= -χ .* Diff(T[2:end-1, 2:end-1]; dims=1) ./ dx
            qTy[:, 2:end-1] .= -χ .* Diff(T[2:end-1, 2:end-1]; dims=2) ./ dy
            T[2:end-1, 2:end-1] .= T_old[2:end-1, 2:end-1] .+ dt .* (.-Diff(qTx; dims=1) ./ dx .- Diff(qTy; dims=2) ./ dy .+ (1 / C) .* av4(τII .^ 2 ./ μ))
            T[[1, end], :] .= T[[2, end - 1], :]
            T[:, [1, end]] .= T[:, [2, end - 1]]
            if iter % ncheck == 0
                err_Pr = maximum(abs.(∇V)) / (σ0^npow * A * exp(-E_R / T0))
                err_Vx = maximum(abs.(-Diff(Pr; dims=1) ./ dx .+ Diff(τxx; dims=1) ./ dx .+ Diff(τxy[2:end-1, :]; dims=2) ./ dy)) / σ0 * h
                err_Vy = maximum(abs.(-Diff(Pr; dims=2) ./ dy .+ Diff(τyy; dims=2) ./ dy .+ Diff(τxy[:, 2:end-1]; dims=1) ./ dx)) / σ0 * h
                @printf("  iter / nx = %.1f, err: [Pr = %1.3e, Vx = %1.3e; Vy = %1.3e]\n", iter / nx, err_Pr, err_Vx, err_Vy)
                if !isfinite(err_Pr) || !isfinite(err_Vx) || !isfinite(err_Vy)
                    error("simulation failed")
                end
                if max(err_Pr, err_Vx, err_Vy) < ϵtol
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
        push!(σ_evo, maximum(τII) / σ0)
        # convergence check
        if any(.!isfinite.(T)) || any(.!isfinite.(τII))
            error("simulation failed")
        end
        # visualisation
        if it % nvis == 0
            # plots
            plts.T[3]        = T[2:end-1, 2:end-1] ./ E_R
            plts.T_sl[1]     = Point2.(xc ./ h, T[2:end-1, ny÷2] ./ E_R)
            plts.Vy_sl[1]    = Point2.(xc ./ h, Vy[2:end-1, ny÷2])
            plts.σ_evo[1]    = Point2.(time_evo, σ_evo)
            plts.Tmax_evo[1] = Point2.(time_evo, Tmax_evo)
            plts.Vmax_evo[1] = Point2.(time_evo, Vmax_evo)
            # autolimits!(axs.T)
            autolimits!(axs.Vy_sl)
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
