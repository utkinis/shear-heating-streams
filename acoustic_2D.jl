using KernelAbstractions
const KA = KernelAbstractions

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

@kernel function init_Pr!(Pr, Lw, xc, yc)
    ix, iy = @index(Global, NTuple)
    @inbounds if ix <= size(Pr, 1) && iy <= size(Pr, 2)
        Pr[ix, iy] = exp(-(xc[ix] / Lw)^2 - (yc[iy] / Lw)^2)
    end
end

function main(backend)
    # physics
    Lx, Ly = 1.0, 1.0
    Lw = 0.1Lx
    K = 1.0
    rho = 1.0
    # numerics
    nx, ny = 128, 128
    nt     = nx
    # preprocessing
    dx, dy = Lx/nx, Ly/ny
    xc = LinRange(-Lx/2 + dx/2, Lx/2 - dx/2, nx)
    yc = LinRange(-Ly/2 + dy/2, Ly/2 - dy/2, ny)
    # parameters
    dt = dx / sqrt(K / rho) / 2.0
    Kdt = K * dt
    dt_rho = dt / rho
    # array allocation
    Pr = KA.zeros(backend, Float64, nx, ny)
    Vx = KA.zeros(backend, Float64, nx + 1, ny)
    Vy = KA.zeros(backend, Float64, nx, ny + 1)
    # init
    init_Pr!(backend, (32, 8), (nx, ny))(Pr, Lw, xc, yc)
    KA.synchronize(backend)

    Pr_ini = Array(Pr)
    # action

    ttot = @elapsed for it in 1:nt
        @info "it" it
        update_stress!(backend, (32, 8), (nx, ny))(Pr, Vx, Vy, Kdt, dx, dy)
        update_velocity!(backend, (32, 8), (nx + 1, ny + 1))(Vx, Vy, Pr, dt_rho, dx, dy)
        KA.synchronize(backend)
    end

    GBs = 2 * (sizeof(Pr) + sizeof(Vx) + sizeof(Vy)) / ttot / 1e9 * nt

    println("time = $ttot s, bandwidth = $GBs GB/s")

    open(io -> write(io, Lx, Ly, dx, dy), "dparams.dat", "w")
    open(io -> write(io, nx, ny), "iparams.dat", "w")
    
    Pr_res = Array(Pr)
    open("Pr.dat", "w") do io
        write(io, Pr_ini)
        write(io, Pr_res)
    end

    return
end

main(CPU())
# main(CUDABackend())