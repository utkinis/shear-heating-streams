using CairoMakie
using Printf

using CUDA
CUDA.allowscalar(false)
CUDA.device!(5)

using KernelAbstractions

@views av1(a) = 0.5 .* (a[1:end-1] .+ a[2:end])
@views avx(a) = 0.5 .* (a[1:end-1, :] .+ a[2:end, :])
@views avy(a) = 0.5 .* (a[:, 1:end-1] .+ a[:, 2:end])
@views av4(a) = 0.25 .* (a[1:end-1, 1:end-1] .+ a[2:end, 1:end-1] .+ a[2:end, 2:end] .+ a[1:end-1, 2:end])

@kernel function update_σ!(Pr, τxx, τyy, τxy, τxx_old, τyy_old, τxy_old, Vx, Vy, μ, G, dt, dτ_r, r, θ_dτ, dx, dy)
    ix, iy = @index(Global, NTuple)
    @inbounds if checkbounds(Bool, Pr, ix, iy)
        ε̇xx = (Vx[ix+1, iy+1] - Vx[ix, iy+1]) / dx
        ε̇yy = (Vy[ix+1, iy+1] - Vy[ix+1, iy]) / dy
        ∇V   = ε̇xx + ε̇yy
        μ_av = 0.25 * (μ[ix, iy] + μ[ix+1, iy] + μ[ix+1, iy+1] + μ[ix, iy+1])
        μ_ve = 1 / (1 / (G * dt) + 1 / μ_av)
        # update pressure
        Pr[ix, iy] -= ∇V * μ_ve * (r / θ_dτ)
        # diagonal deviatoric stress
        dτxx_dt = (τxx[ix, iy] - τxx_old[ix, iy]) / dt
        dτyy_dt = (τyy[ix, iy] - τyy_old[ix, iy]) / dt
        τxx[ix, iy] += (-dτxx_dt / G - τxx[ix, iy] / μ_av + 2.0 * (ε̇xx - ∇V / 3.0)) * μ_ve * dτ_r
        τyy[ix, iy] += (-dτyy_dt / G - τyy[ix, iy] / μ_av + 2.0 * (ε̇yy - ∇V / 3.0)) * μ_ve * dτ_r
    end
    @inbounds if checkbounds(Bool, τxy, ix, iy)
        ε̇xy = 0.5 * ((Vx[ix, iy+1] - Vx[ix, iy]) / dy + (Vy[ix+1, iy] - Vy[ix, iy]) / dx)
        μ_ve = 1 / (1 / (G * dt) + 1 / μ[ix, iy])
        dτxy_dt = (τxy[ix, iy] - τxy_old[ix, iy]) / dt
        τxy[ix, iy] += (-dτxy_dt / G - τxy[ix, iy] / μ[ix, iy] + 2.0 * ε̇xy) * μ_ve * dτ_r
    end
end

@kernel function compute_τII!(τII, τxx, τyy, τxy)
    ix, iy = @index(Global, NTuple)
    @inbounds begin
        τxx_av = 0.25 * (τxx[ix, iy] + τxx[ix+1, iy] + τxx[ix+1, iy+1] + τxx[ix, iy+1])
        τyy_av = 0.25 * (τyy[ix, iy] + τyy[ix+1, iy] + τyy[ix+1, iy+1] + τyy[ix, iy+1])
        τII[ix+1, iy+1] = sqrt(0.5 * (τxx_av^2 + τyy_av^2) + τxy[ix+1, iy+1]^2)
    end
end

@kernel function compute_μ!(μ, τII, T, A, E_R, npow)
    ix, iy = @index(Global, NTuple)
    @inbounds begin
        T_av = 0.25 * (T[ix, iy] + T[ix+1, iy] + T[ix+1, iy+1] + T[ix, iy+1])
        μ[ix, iy] = (1 / A) * exp(E_R / T_av) * τII[ix, iy]^(1 - npow)
    end
end

@kernel function update_V!(Vx, Vy, Pr, τxx, τyy, τxy, μ, G, dt, nudτ, dx, dy)
    ix, iy = @index(Global, NTuple)
    @inbounds if ix <= size(Vx, 1) - 2 && iy <= size(Vx, 2) - 2
        μ_mloc = max(μ[ix+0, iy+0], μ[ix+1, iy+0], μ[ix+2, iy+0],
                     μ[ix+0, iy+1], μ[ix+1, iy+1], μ[ix+2, iy+1])
        ∂σxx_∂x = -(Pr[ix+1, iy] - Pr[ix, iy]) / dx + (τxx[ix+1, iy] - τxx[ix, iy]) / dx
        ∂τxy_∂y = (τxy[ix+1, iy+1] - τxy[ix+1, iy]) / dy
        Vx[ix+1, iy+1] += (∂σxx_∂x + ∂τxy_∂y) * (1 / (G * dt) + 1 / μ_mloc) * nudτ
    end
    @inbounds if ix <= size(Vy, 1) - 2 && iy <= size(Vy, 2) - 2
        μ_mloc = max(μ[ix+0, iy+0], μ[ix+0, iy+1], μ[ix+0, iy+2],
                     μ[ix+1, iy+0], μ[ix+1, iy+1], μ[ix+1, iy+2])
        ∂σyy_∂y = -(Pr[ix, iy+1] - Pr[ix, iy]) / dy + (τyy[ix, iy+1] - τyy[ix, iy]) / dy
        ∂τxy_∂x = (τxy[ix+1, iy+1] - τxy[ix, iy+1]) / dx
        Vy[ix+1, iy+1] += (∂σyy_∂y + ∂τxy_∂x) * (1 / (G * dt) + 1 / μ_mloc) * nudτ
    end
end

@kernel function compute_qT!(qTx, qTy, T, χ, dx, dy)
    ix, iy = @index(Global, NTuple)
    @inbounds if ix <= size(qTx, 1) - 2 && iy <= size(qTx, 2)
        qTx[ix+1, iy] = -χ * (T[ix+2, iy+1] - T[ix+1, iy+1]) / dx
    end
    @inbounds if ix <= size(qTy, 1) && iy <= size(qTy, 2) - 2
        qTy[ix, iy+1] = -χ * (T[ix+1, iy+2] - T[ix+1, iy+1]) / dy
    end
end

@kernel function update_T!(T, T_old, qTx, qTy, τII, μ, C, dt, dx, dy)
    ix, iy = @index(Global, NTuple)
    @inbounds begin
        sh = 0.25 * (τII[ix+0, iy+0]^2 / μ[ix+0, iy+0] +
                     τII[ix+1, iy+0]^2 / μ[ix+1, iy+0] +
                     τII[ix+1, iy+1]^2 / μ[ix+1, iy+1] +
                     τII[ix+0, iy+1]^2 / μ[ix+0, iy+1])
        divqT = (qTx[ix+1, iy] - qTx[ix, iy]) / dx + (qTy[ix, iy+1] - qTy[ix, iy]) / dy
        T[ix+1, iy+1] = T_old[ix+1, iy+1] + dt * (-divqT + (1 / C) * sh)
    end
end

@kernel function neumann_bc_x!(A)
    iy = @index(Global, Linear)
    @inbounds A[1, iy] = A[2, iy]
    @inbounds A[end, iy] = A[end-1, iy]
end

@kernel function neumann_bc_y!(A)
    ix = @index(Global, Linear)
    @inbounds A[ix, 1] = A[ix, 2]
    @inbounds A[ix, end] = A[ix, end-1]
end

@kernel function dirichlet_bc_x!(A)
    iy = @index(Global, Linear)
    @inbounds A[1, iy] = -A[2, iy]
    @inbounds A[end, iy] = -A[end-1, iy]
end

@kernel function dirichlet_bc_y!(A)
    ix = @index(Global, Linear)
    @inbounds A[ix, 1] = -A[ix, 2]
    @inbounds A[ix, end] = -A[ix, end-1]
end

@views function braeck_2D(backend)
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
    nx     = 511
    ny     = 511
    niter  = 20min(nx, ny)
    nvis   = 5
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
    T     = KernelAbstractions.zeros(backend, Float64, nx + 2, ny + 2)
    T_old = KernelAbstractions.zeros(backend, Float64, nx + 2, ny + 2)
    qTx   = KernelAbstractions.zeros(backend, Float64, nx + 1, ny)
    qTy   = KernelAbstractions.zeros(backend, Float64, nx, ny + 1)
    # mechanics
    Pr      = KernelAbstractions.zeros(backend, Float64, nx, ny)
    τxx     = KernelAbstractions.zeros(backend, Float64, nx, ny)
    τyy     = KernelAbstractions.zeros(backend, Float64, nx, ny)
    τxy     = KernelAbstractions.zeros(backend, Float64, nx + 1, ny + 1)
    τII     = KernelAbstractions.zeros(backend, Float64, nx + 1, ny + 1)
    τxx_old = KernelAbstractions.zeros(backend, Float64, nx, ny)
    τyy_old = KernelAbstractions.zeros(backend, Float64, nx, ny)
    τxy_old = KernelAbstractions.zeros(backend, Float64, nx + 1, ny + 1)
    Vx      = KernelAbstractions.zeros(backend, Float64, nx + 1, ny + 2)
    Vy      = KernelAbstractions.zeros(backend, Float64, nx + 2, ny + 1)
    ∇V      = KernelAbstractions.zeros(backend, Float64, nx, ny)
    μ       = KernelAbstractions.zeros(backend, Float64, nx + 1, ny + 1)
    # initialisation
    broadcast!(T[2:end-1, 2:end-1], xc, yc') do x, y
        abs(x) <= h / 2 ? T0 : Tbg
    end
    T[[1, end], :] .= T[[2, end - 1], :]
    T[:, [1, end]] .= T[:, [2, end - 1]]
    τxy .= σ0
    τII[2:end-1, 2:end-1] .= sqrt.(0.5 .* (av4(τxx) .^ 2 .+ av4(τyy) .^ 2) .+ τxy[2:end-1, 2:end-1] .^ 2)
    τII[[1, end], :] .= τII[[2, end - 1], :]
    τII[:, [1, end]] .= τII[:, [2, end - 1]]
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
    plts = (T        = heatmap!(axs.T, xc, yc, Array(T[2:end-1, 2:end-1] ./ E_R); colormap=:turbo),
            T_ini_sl = lines!(axs.T_sl, Point2.(xc ./ h, Array(T[2:end-1, ny÷2] ./ E_R)); linewidth=4),
            T_sl     = lines!(axs.T_sl, Point2.(xc ./ h, Array(T[2:end-1, ny÷2] ./ E_R)); linewidth=4),
            Tmaxa    = hlines!(axs.T_sl, Tmaxa; linewidth=4, color=:gray, linestyle=:dash),
            Vy_sl    = lines!(axs.Vy_sl, Point2.(xc ./ h, Array(Vy[2:end-1, ny÷2])); linewidth=4),
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
        dσ_dt = maximum(abs.(-(A * G) .* τII .^ npow .* exp.(-E_R ./ av4(T)) .+ G .* (diff(Vx; dims=2) ./ dy .+ diff(Vy; dims=1) ./ dx)))
        dt = min(1e-4 * σ0 / dσ_dt, dt_diff, 1e-3 * τr)
        # iteration loop
        for iter in 1:niter
            # stress
            compute_τII!(backend, 256, (nx - 1, ny - 1))(τII, τxx, τyy, τxy)
            neumann_bc_x!(backend, 256, ny + 1)(τII)
            neumann_bc_y!(backend, 256, nx + 1)(τII)
            compute_μ!(backend, 256, (nx + 1, ny + 1))(μ, τII, T, A, E_R, npow)
            update_σ!(backend, 256, (nx + 1, ny + 1))(Pr, τxx, τyy, τxy, τxx_old, τyy_old, τxy_old, Vx, Vy, μ, G, dt, dτ_r, r, θ_dτ, dx, dy)
            # velocity
            update_V!(backend, 256, (nx, ny))(Vx, Vy, Pr, τxx, τyy, τxy, μ, G, dt, nudτ, dx, dy)
            dirichlet_bc_y!(backend, 256, nx + 2)(Vx)
            dirichlet_bc_x!(backend, 256, ny + 2)(Vy)
            # temperature
            compute_qT!(backend, 256, (nx, ny))(qTx, qTy, T, χ, dx, dy)
            update_T!(backend, 256, (nx, ny))(T, T_old, qTx, qTy, τII, μ, C, dt, dx, dy)
            neumann_bc_x!(backend, 256, ny + 2)(T)
            neumann_bc_y!(backend, 256, nx + 2)(T)
            if iter % ncheck == 0
                ∇V .= diff(Vx[:, 2:end-1]; dims=1) ./ dx .+ diff(Vy[2:end-1, :]; dims=2) ./ dy
                err_Pr = maximum(abs.(∇V)) / (σ0^npow * A * exp(-E_R / T0))
                err_Vx = maximum(abs.(.-diff(Pr; dims=1) ./ dx .+ diff(τxx; dims=1) ./ dx .+ diff(τxy[2:end-1, :]; dims=2) ./ dy)) / σ0 * h
                err_Vy = maximum(abs.(.-diff(Pr; dims=2) ./ dy .+ diff(τyy; dims=2) ./ dy .+ diff(τxy[:, 2:end-1]; dims=1) ./ dx)) / σ0 * h
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
            plts.T[3]        = Array(T[2:end-1, 2:end-1] ./ E_R)
            plts.T_sl[1]     = Point2.(xc ./ h, Array(T[2:end-1, ny÷2] ./ E_R))
            plts.Vy_sl[1]    = Point2.(xc ./ h, Array(Vy[2:end-1, ny÷2]))
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
            # yield()
            display(fig)
        end
    end
    @show (maximum(Tmax_evo) - T0_E_R) / (Tmaxa - T0_E_R)
    # run(`ffmpeg -framerate 15 -i anim/step_%04d.png -c libx264 -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2:color=white" -y adiabatic.mp4`)
    return
end

braeck_2D(CUDABackend())
