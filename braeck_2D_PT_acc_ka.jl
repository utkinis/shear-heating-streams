using OffsetArrays, Printf, KernelAbstractions, CairoMakie

using CUDA
CUDA.allowscalar(false)
CUDA.device!(5)

@views av1(a) = 0.5 .* (a[begin:end-1] .+ a[begin+1:end])
@views avx(a) = 0.5 .* (a[begin:end-1, :] .+ a[begin+1:end, :])
@views avy(a) = 0.5 .* (a[:, begin:end-1] .+ a[:, begin+1:end])
@views av4(a) = 0.25 .* (a[begin+0:end-1, begin+0:end-1] .+
                         a[begin+1:end+0, begin+0:end-1] .+
                         a[begin+1:end+0, begin+1:end+0] .+
                         a[begin+0:end-1, begin+1:end+0])

@kernel function update_σ!(Pr, τ, τ_old, V, μ, G, dt, dτ_r, r, θ_dτ, dx, dy, nx, ny)
    ix, iy = @index(Global, NTuple)
    @inbounds if ix <= nx && iy <= ny
        ε̇xx = (V.x[ix+1, iy] - V.x[ix, iy]) / dx
        ε̇yy = (V.y[ix, iy+1] - V.y[ix, iy]) / dy
        ∇V   = ε̇xx + ε̇yy
        μ_c  = 0.25 * (μ[ix, iy] + μ[ix+1, iy] + μ[ix+1, iy+1] + μ[ix, iy+1])
        μ_ve = 1 / (1 / (G * dt) + 1 / μ_c)
        # update pressure
        Pr[ix, iy] -= ∇V * μ_ve * (r / θ_dτ)
        # diagonal deviatoric stress
        dτxx_dt = (τ.xx[ix, iy] - τ_old.xx[ix, iy]) / dt
        dτyy_dt = (τ.yy[ix, iy] - τ_old.yy[ix, iy]) / dt
        τ.xx[ix, iy] += (-dτxx_dt / G - τ.xx[ix, iy] / μ_c + 2.0 * (ε̇xx - ∇V / 3.0)) * μ_ve * dτ_r
        τ.yy[ix, iy] += (-dτyy_dt / G - τ.yy[ix, iy] / μ_c + 2.0 * (ε̇yy - ∇V / 3.0)) * μ_ve * dτ_r
    end
    @inbounds if ix <= nx + 1 && iy <= ny + 1
        ε̇xy = 0.5 * ((V.x[ix, iy] - V.x[ix, iy-1]) / dy + (V.y[ix, iy] - V.y[ix-1, iy]) / dx)
        μ_ve = 1 / (1 / (G * dt) + 1 / μ[ix, iy])
        dτxy_dt = (τ.xy[ix, iy] - τ_old.xy[ix, iy]) / dt
        τ.xy[ix, iy] += (-dτxy_dt / G - τ.xy[ix, iy] / μ[ix, iy] + 2.0 * ε̇xy) * μ_ve * dτ_r
    end
end

@kernel function compute_τII!(τII, τ)
    ix, iy = @index(Global, NTuple)
    @inbounds begin
        τxx_av = 0.25 * (τ.xx[ix-1, iy-1] + τ.xx[ix, iy-1] + τ.xx[ix, iy] + τ.xx[ix-1, iy])
        τyy_av = 0.25 * (τ.yy[ix-1, iy-1] + τ.yy[ix, iy-1] + τ.yy[ix, iy] + τ.yy[ix-1, iy])
        τII[ix, iy] = sqrt(0.5 * (τxx_av^2 + τyy_av^2) + τ.xy[ix, iy]^2)
    end
end

@kernel function compute_μ!(μ, τII, T, A, E_R, npow)
    ix, iy = @index(Global, NTuple)
    @inbounds begin
        T_av = 0.25 * (T[ix-1, iy-1] + T[ix, iy-1] + T[ix, iy] + T[ix-1, iy])
        μ[ix, iy] = (1 / A) * exp(E_R / T_av) * τII[ix, iy]^(1 - npow)
    end
end

@kernel function update_V!(V, Pr, τ, μ, G, dt, nudτ, dx, dy, nx, ny)
    ix, iy = @index(Global, NTuple)
    @inbounds if ix <= nx + 1 && iy <= ny
        μ_mloc = max(μ[ix-1, iy+0], μ[ix, iy+0], μ[ix+1, iy+0],
                     μ[ix-1, iy+1], μ[ix, iy+1], μ[ix+1, iy+1])
        ∂σxx_∂x = -(Pr[ix, iy] - Pr[ix-1, iy]) / dx + (τ.xx[ix, iy] - τ.xx[ix-1, iy]) / dx
        ∂τxy_∂y = (τ.xy[ix, iy+1] - τ.xy[ix, iy]) / dy
        V.x[ix, iy] += (∂σxx_∂x + ∂τxy_∂y) * (1 / (G * dt) + 1 / μ_mloc) * nudτ
    end
    @inbounds if ix <= nx && iy <= ny + 1
        μ_mloc = max(μ[ix+0, iy-1], μ[ix+0, iy], μ[ix+0, iy+1],
                     μ[ix+1, iy-1], μ[ix+1, iy], μ[ix+1, iy+1])
        ∂σyy_∂y = -(Pr[ix, iy] - Pr[ix, iy-1]) / dy + (τ.yy[ix, iy] - τ.yy[ix, iy-1]) / dy
        ∂τxy_∂x = (τ.xy[ix+1, iy] - τ.xy[ix, iy]) / dx
        V.y[ix, iy] += (∂σyy_∂y + ∂τxy_∂x) * (1 / (G * dt) + 1 / μ_mloc) * nudτ
    end
end

@kernel function compute_residuals!(res_Pr, res_V, Pr, τ, V, dx, dy, nx, ny)
    ix, iy = @index(Global, NTuple)
    @inbounds if ix <= nx && iy <= ny
        ε̇xx = (V.x[ix+1, iy] - V.x[ix, iy]) / dx
        ε̇yy = (V.y[ix, iy+1] - V.y[ix, iy]) / dy
        res_Pr[ix, iy] = ε̇xx + ε̇yy
    end
    @inbounds if ix <= nx + 1 && iy <= ny
        ∂σxx_∂x = -(Pr[ix, iy] - Pr[ix-1, iy]) / dx + (τ.xx[ix, iy] - τ.xx[ix-1, iy]) / dx
        ∂τxy_∂y = (τ.xy[ix, iy+1] - τ.xy[ix, iy]) / dy
        res_V.x[ix, iy] = ∂σxx_∂x + ∂τxy_∂y
    end
    @inbounds if ix <= nx && iy <= ny + 1
        ∂σyy_∂y = -(Pr[ix, iy] - Pr[ix, iy-1]) / dy + (τ.yy[ix, iy] - τ.yy[ix, iy-1]) / dy
        ∂τxy_∂x = (τ.xy[ix+1, iy] - τ.xy[ix, iy]) / dx
        res_V.y[ix, iy] = ∂σyy_∂y + ∂τxy_∂x
    end
end

@kernel function init_T!(T, Tbg, T0, xc, yc, h)
    ix, iy = @index(Global, NTuple)
    @inbounds T[ix, iy] = abs(xc[ix]) <= h / 2 ? T0 : Tbg
end

@kernel function compute_qT!(qT, T, χ, dx, dy, nx, ny)
    ix, iy = @index(Global, NTuple)
    @inbounds if ix <= nx + 1 && iy <= ny
        qT.x[ix, iy] = -χ * (T[ix, iy] - T[ix-1, iy]) / dx
    end
    @inbounds if ix <= nx && iy <= ny + 1
        qT.y[ix, iy] = -χ * (T[ix, iy] - T[ix, iy-1]) / dy
    end
end

@kernel function update_T!(T, T_old, qT, τII, μ, C, dt, dx, dy)
    ix, iy = @index(Global, NTuple)
    @inbounds begin
        sh = 0.25 * (τII[ix+0, iy+0]^2 / μ[ix+0, iy+0] +
                     τII[ix+1, iy+0]^2 / μ[ix+1, iy+0] +
                     τII[ix+1, iy+1]^2 / μ[ix+1, iy+1] +
                     τII[ix+0, iy+1]^2 / μ[ix+0, iy+1])
        divqT = (qT.x[ix+1, iy] - qT.x[ix, iy]) / dx + (qT.y[ix, iy+1] - qT.y[ix, iy]) / dy
        T[ix, iy] = T_old[ix, iy] + dt * (-divqT + (1 / C) * sh)
    end
end

@kernel function neumann_bc_x!(A)
    iy = @index(Global, Linear)
    @inbounds A[begin, iy] = A[begin+1, iy]
    @inbounds A[end, iy] = A[end-1, iy]
end

@kernel function neumann_bc_y!(A)
    ix = @index(Global, Linear)
    @inbounds A[ix, begin] = A[ix, begin+1]
    @inbounds A[ix, end] = A[ix, end-1]
end

@kernel function dirichlet_bc_x!(A)
    iy = @index(Global, Linear)
    @inbounds A[begin, iy] = -A[begin+1, iy]
    @inbounds A[end, iy] = -A[end-1, iy]
end

@kernel function dirichlet_bc_y!(A)
    ix = @index(Global, Linear)
    @inbounds A[ix, begin] = -A[ix, begin+1]
    @inbounds A[ix, end] = -A[ix, end-1]
end

function scalar_field(backend, type, nx, ny)
    field = KernelAbstractions.zeros(backend, type, nx + 2, ny + 2)
    return OffsetArray(field, -1, -1)
end

function vector_field(backend, type, nx, ny)
    return (x = scalar_field(backend, type, nx + 1, ny),
            y = scalar_field(backend, type, nx, ny + 1))
end

function tensor_field(backend, type, nx, ny)
    return (xx = scalar_field(backend, type, nx, ny),
            yy = scalar_field(backend, type, nx, ny),
            xy = scalar_field(backend, type, nx + 1, ny + 1))
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
    T     = scalar_field(backend, Float64, nx, ny)
    T_old = scalar_field(backend, Float64, nx, ny)
    qT    = vector_field(backend, Float64, nx, ny)
    # mechanics
    Pr    = scalar_field(backend, Float64, nx, ny)
    τ     = tensor_field(backend, Float64, nx, ny)
    τ_old = tensor_field(backend, Float64, nx, ny)
    τII   = scalar_field(backend, Float64, nx + 1, ny + 1)
    V     = vector_field(backend, Float64, nx, ny)
    μ     = scalar_field(backend, Float64, nx + 1, ny + 1)
    # residuals
    res_Pr = scalar_field(backend, Float64, nx, ny)
    res_V  = vector_field(backend, Float64, nx, ny)
    # initialisation
    init_T!(backend, 256, (nx, ny))(T, Tbg, T0, xc, yc, h)
    neumann_bc_x!(backend, 256, ny + 2)(T)
    neumann_bc_y!(backend, 256, nx + 2)(T)
    τ.xy .= σ0
    compute_τII!(backend, 256, (nx + 1, ny + 1))(τII, τ)
    neumann_bc_x!(backend, 256, ny + 3)(τII)
    neumann_bc_y!(backend, 256, nx + 3)(τII)
    # temporal evolution
    time_evo = Float64[0.0]
    Tmax_evo = Float64[maximum(parent(T))/E_R]
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
    plts = (T        = heatmap!(axs.T, xc, yc, Array(parent(T)[2:end-1, 2:end-1] ./ E_R); colormap=:turbo),
            T_ini_sl = lines!(axs.T_sl, Point2.(xc ./ h, Array(parent(T)[2:end-1, ny÷2] ./ E_R)); linewidth=4),
            T_sl     = lines!(axs.T_sl, Point2.(xc ./ h, Array(parent(T)[2:end-1, ny÷2] ./ E_R)); linewidth=4),
            Tmaxa    = hlines!(axs.T_sl, Tmaxa; linewidth=4, color=:gray, linestyle=:dash),
            Vy_sl    = lines!(axs.Vy_sl, Point2.(xc ./ h, Array(parent(V.y)[2:end-1, ny÷2])); linewidth=4),
            σ_evo    = lines!(axs.σ_evo, Point2.(time_evo, maximum(parent(τ.xy)) / σ0); linewidth=4),
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
        copyto!(parent(τ_old.xx), parent(τ.xx))
        copyto!(parent(τ_old.yy), parent(τ.yy))
        copyto!(parent(τ_old.xy), parent(τ.xy))
        copyto!(parent(T_old   ), parent(T))
        # time step
        dσ_dt = maximum(abs.(-(A * G) .* parent(τII)[2:end-1,2:end-1] .^ npow .* exp.(-E_R ./ av4(parent(T))) .+ G .* (diff(parent(V.x)[2:end-1,:]; dims=2) ./ dy .+ diff(parent(V.y)[:,2:end-1]; dims=1) ./ dx)))
        dt = min(1e-4 * σ0 / dσ_dt, dt_diff, 1e-3 * τr)
        # iteration loop
        for iter in 1:niter
            # stress
            compute_τII!(backend, 256, (nx + 1, ny + 1))(τII, τ)
            neumann_bc_x!(backend, 256, ny + 3)(τII)
            neumann_bc_y!(backend, 256, nx + 3)(τII)
            compute_μ!(backend, 256, (nx + 1, ny + 1))(μ, τII, T, A, E_R, npow)
            update_σ!(backend, 256, (nx + 1, ny + 1))(Pr, τ, τ_old, V, μ, G, dt, dτ_r, r, θ_dτ, dx, dy, nx, ny)
            # velocity
            update_V!(backend, 256, (nx, ny))(V, Pr, τ, μ, G, dt, nudτ, dx, dy, nx, ny)
            parent(V.x)[[2, nx + 2], :] .= 0.0
            parent(V.y)[:, [2, ny + 2]] .= 0.0
            dirichlet_bc_x!(backend, 256, ny + 2)(V.y)
            dirichlet_bc_y!(backend, 256, nx + 2)(V.x)
            # temperature
            compute_qT!(backend, 256, (nx + 1, ny + 1))(qT, T, χ, dx, dy, nx, ny)
            update_T!(backend, 256, (nx, ny))(T, T_old, qT, τII, μ, C, dt, dx, dy)
            neumann_bc_x!(backend, 256, ny + 2)(T)
            neumann_bc_y!(backend, 256, nx + 2)(T)
            if iter % ncheck == 0
                compute_residuals!(backend, 256, (nx + 1, ny + 1))(res_Pr, res_V, Pr, τ, V, dx, dy, nx, ny)
                parent(res_V.x)[[2, nx + 2], :] .= 0.0
                parent(res_V.y)[:, [2, ny + 2]] .= 0.0
                err_Pr = maximum(abs.(parent(res_Pr))) / (σ0^npow * A * exp(-E_R / T0))
                err_Vx = maximum(abs.(parent(res_V.x))) / σ0 * h
                err_Vy = maximum(abs.(parent(res_V.y))) / σ0 * h
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
        push!(Tmax_evo, maximum(parent(T)) / E_R)
        push!(Vmax_evo, maximum(parent(V.y)) / (σ0^npow * h * A * exp(-E_R / T0)))
        push!(σ_evo, maximum(parent(τII)) / σ0)
        # convergence check
        if any(.!isfinite.(parent(T))) || any(.!isfinite.(parent(τII)))
            error("simulation failed")
        end
        # visualisation
        if it % nvis == 0
            # plots
            plts.T[3]        = Array(parent(T)[2:end-1, 2:end-1] ./ E_R)
            plts.T_sl[1]     = Point2.(xc ./ h, Array(parent(T)[2:end-1, ny÷2] ./ E_R))
            plts.Vy_sl[1]    = Point2.(xc ./ h, Array(parent(V.y)[2:end-1, ny÷2]))
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
