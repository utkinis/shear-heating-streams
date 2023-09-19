using KernelAbstractions
const KA = KernelAbstractions

# using CUDA
# using AMDGPU

@kernel function update_stress!(Pr, Vx, Vy, Kdt, dx, dy)
    ix, iy = @index(Global, NTuple)
    @inbounds if ix <= size(Pr, 1) && iy <= size(Pr, 2)
        exx = (Vx[ix + 1, iy] - Vx[ix, iy]) / dx
        eyy = (Vy[ix, iy + 1] - Vy[ix, iy]) / dy
        divV = exx + eyy
        Pr[ix, iy] -= divV * Kdt
    end
end

@kernel function update_velocity!(Vx, Vy, Pr, dt_rho, dx, dy)
    ix, iy = @index(Global, NTuple)
    @inbounds if 1 < ix < size(Vx, 1) && iy <= size(Vx, 2)
        Vx[ix, iy] += -dt_rho * (Pr[ix, iy] - Pr[ix - 1, iy]) / dx
    end
    @inbounds if ix <= size(Vy, 1) && 1 < iy < size(Vy, 2)
        Vy[ix, iy] += -dt_rho * (Pr[ix, iy] - Pr[ix, iy - 1]) / dy
    end
end

@kernel function init_Pr!(Pr, Lw, xc, yc)
    ix, iy = @index(Global, NTuple)
    @inbounds if ix <= size(Pr, 1) && iy <= size(Pr, 2)
        Pr[ix, iy] = exp(-(xc[ix] / Lw)^2 - (yc[iy] / Lw)^2)
    end
end

function main(backend)
    # remove .dat files
    rm("out"; force=true, recursive=true)
    mkdir("out")
    # physics
    Lx, Ly = 1.0, 1.0
    Lw     = 0.1Lx
    K      = 1.0
    rho    = 1.0
    # numerics
    nx, ny     = 512 - 1, 512 - 1
    nsave      = 100
    nt         = 10nsave
    save_steps = true
    # preprocessing
    dx, dy = Lx / nx, Ly / ny
    xc = LinRange(-Lx / 2 + dx / 2, Lx / 2 - dx / 2, nx)
    yc = LinRange(-Ly / 2 + dy / 2, Ly / 2 - dy / 2, ny)
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
    # write parameters
    write("out/dparams.dat", Lx, Ly, dx, dy)
    write("out/iparams.dat", nx, ny, nt, nsave)
    write("out/step_0.dat", Array(Pr))
    # action
    ttot = @elapsed begin
        for it in 1:nt
            update_stress!(backend, (32, 8), (nx, ny))(Pr, Vx, Vy, Kdt, dx, dy)
            update_velocity!(backend, (32, 8), (nx + 1, ny + 1))(Vx, Vy, Pr, dt_rho, dx, dy)
            if save_steps && it % nsave == 0
                @info "save" it
                KA.synchronize(backend)
                write("out/step_$it.dat", Array(Pr))
            end
        end
        KA.synchronize(backend)
    end
    if !save_steps
        write("out/step_$nt.dat", Array(Pr))
    end
    # calculate memory throughput
    size_rw = sizeof(Pr) + sizeof(Vx) + sizeof(Vy)
    GBs     = (2 * size_rw) / ttot / 1e9 * nt
    println("time = $ttot s, bandwidth = $GBs GB/s")
    return
end

main(CPU())
# main(CUDABackend())
# main(ROCBackend())