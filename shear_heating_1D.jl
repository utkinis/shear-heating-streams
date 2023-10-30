using Printf
using LazyArrays
using GLMakie
Makie.inline!(false)

@views avx(a) = 0.5 .* (a[1:end-1] .+ a[2:end])

@views function shear_heating_1D()
    ## physics
    lx   = 1.0
    ηs0  = 100.0
    Gs   = 1.0
    ρs   = 1.0
    λ    = 1.0e-6
    cp   = 1.0
    E_R  = 2.0
    T_bg = 1.0
    ΔT   = 1.0
    τ0   = 5.0
    ## scales
    tsc = ηs0 / Gs
    ## numerics
    nx = 200
    nt = 100000
    nvis = 10
    niter = 20nx^2
    ncheck = ceil(Int, 0.25nx^2)
    ϵtol = 1e-8
    ## preprocessing
    dx = lx / nx
    xv = LinRange(-lx / 2, lx / 2, nx + 1)
    xc = avx(xv)
    dt = min(dx^2 / (λ / (ρs * cp)) / 4.1, 1e-3tsc)
    ## array alloc
    Vy   = zeros(nx + 2)
    τxy  = zeros(nx + 1)
    ηs   = zeros(nx + 1)
    dτ_ρ = zeros(nx)
    T    = zeros(nx + 2)
    qUx  = zeros(nx + 1)
    ## init
    map!(T[2:end-1], xc) do x
        abs(x) < lx / 20 ? T_bg + ΔT : T_bg
    end
    T_old = copy(T)
    τxy .= τ0
    τxy_old = copy(τxy)
    ## vis
    fig = Figure(; resolution=(1000, 1000), fontsize=32)
    axs = (T  = Axis(fig[1, 1]; title="T", xlabel="x"),
           Vy = Axis(fig[2, 1]; title="Vy", xlabel="x"))
    plts = (T  = lines!(axs.T, Point2.(xc, T[2:end-1]); linewidth=2),
            Vy = lines!(axs.Vy, Point2.(xc, Vy[2:end-1]); linewidth=2))
    display(fig)
    ## time loop
    for it in 1:nt
        @printf("it = %d\n", it)
        τxy_old .= τxy
        T_old .= T
        for iter in 1:niter
            ηs .= ηs0 .* exp.(E_R ./ avx(T))
            τxy .= (τxy_old ./ (Gs * dt) .+ Diff(Vy) ./ dx) ./ (1.0 ./ (Gs * dt) .+ 1.0 ./ ηs)
            dτ_ρ .= dx^2 .* (1.0 ./ max.(ηs[1:end-1], ηs[2:end]) .+ 1.0 / (Gs * dt)) ./ 10.1
            Vy[2:end-1] .+= dτ_ρ .* Diff(τxy) ./ dx
            Vy[[1, end]] .= .-Vy[[2, end - 1]]
            qUx .= .-λ .* Diff(T) ./ dx
            T[2:end-1] .= T_old[2:end-1] .+ dt .* (.-Diff(qUx) ./ dx .+ avx(0.5 .* τxy .* Diff(Vy) ./ dx)) ./ (ρs * cp)
            T[[1, end]] .= T[[2, end - 1]]
            if iter % ncheck == 0
                err_Vy = maximum(abs.(Diff(τxy) ./ dx)) / τ0
                @printf("  iter / nx^2 = %.1f, err = %1.3e\n", iter / nx^2, err_Vy)
                if !isfinite(err_Vy)
                    error("simulation failed")
                end
                if err_Vy < ϵtol
                    break
                end
            end
        end
        dt = 1e-3 * minimum(ηs ./ Gs)
        if it % nvis == 0
            println("it = $it")
            plts.T[1] = Point2.(xc, T[2:end-1])
            plts.Vy[1] = Point2.(xc, Vy[2:end-1])
            for ax in axs
                autolimits!(ax)
            end
            yield()
        end
    end
    @show extrema(Vy) dt
    return
end

shear_heating_1D()
