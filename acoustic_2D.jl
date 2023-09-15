using KernelAbstractions
using CUDA

@kernel function update_stress!(Pr, Vx, Vy, Kdt, dx, dy)
    ix, iy = @index(Global, NTuple)
    @inbounds if ix <= size(Pr, 1) && iy <= size(Pr, 2)
        exx = (Vx[ix+1, iy] - Vx[ix, iy]) / dx
        eyy = (Vy[ix, iy+1] - Vy[ix, iy]) / dy
        divV = exx + eyy
        Pr[ix, iy] -= divV * Kdt
    end
end

@kernel function update_velocity!(Vx, Vy, Pr, dt_rho, dx, dy)
    ix, iy = @index(Global, NTuple)
    @inbounds if 1 < ix < size(Vx, 1) && iy <= size(Vx, 2)
        Vx[ix, iy] += -dt_rho * (Pr[ix, iy] - Pr[ix-1, iy]) / dx
    end
    @inbounds if ix <= size(Vy, 1) && 1 < iy < size(Vy, 2)
        Vy[ix, iy] += -dt_rho * (Pr[ix, iy] - Pr[ix, iy-1]) / dy
    end
end

@kernel function init_Pr!(Pr, lw, xc, yc)
    ix, iy = @index(Global, NTuple)
    @inbounds if ix <= size(Pr, 1) && iy <= size(Pr, 2)
        Pr[ix, iy] = exp(-(xc[ix] / lw)^2 - (yc[iy] / lw)^2)
    end
end

function main(backend)
    # physics
    Lx, Ly = 1.0, 1.0
    lw = 0.1Lx
    K = 1.0
    rho = 1.0
    # numerics
    nx, ny = 128, 128
    nt     = nx
    # preprocessing
    dx, dy = Lx / nx, Ly / ny
    xc, yc = LinRange(-Lx / 2 + dx / 2, Lx / 2 - dx / 2, nx), LinRange(-Ly / 2 + dy / 2, Ly / 2 - dy / 2, ny)
    # parameters
    dt = dx / sqrt(K / rho) / 2.0
    Kdt = K * dt
    dt_rho = dt / rho
    # array allocation
    Pr = KernelAbstractions.zeros(backend, Float64, nx, ny)
    Vx = KernelAbstractions.zeros(backend, Float64, nx + 1, ny)
    Vy = KernelAbstractions.zeros(backend, Float64, nx, ny + 1)
    # init
    init_Pr!(backend, (32, 8), (nx, ny))(Pr, lw, xc, yc)
    KernelAbstractions.synchronize(backend)

    Pr_ini = Array(Pr)
    # action

    ttot = @elapsed for it in 1:nt
        @info "it" it
        update_stress!(backend, (32, 8), (nx, ny))(Pr, Vx, Vy, Kdt, dx, dy)
        update_velocity!(backend, (32, 8), (nx + 1, ny + 1))(Vx, Vy, Pr, dt_rho, dx, dy)
        KernelAbstractions.synchronize(backend)
    end

    GBs = 2 * (sizeof(Pr) + sizeof(Vx) + sizeof(Vy)) / ttot / 1e9 * nt

    println("time = $ttot s, bandwidth = $GBs GB/s")

    Pr_h = Array(Pr)
    open("dparams.dat", "w") do io
        write(io, Lx, Ly, dx, dy)
    end

    open("iparams.dat", "w") do io
        write(io, nx, ny)
    end
    open("Pr.dat", "w") do io
        write(io, Pr_ini)
        write(io, Pr_h)
    end
    return
end

main(CPU())
# main(CUDABackend())