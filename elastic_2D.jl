using KernelAbstractions
const KA = KernelAbstractions

# using CUDA
# using AMDGPU

Base.@propagate_inbounds avx(A, ix, iy) = 0.5 * (A[ix, iy] + A[ix + 1, iy])
Base.@propagate_inbounds avy(A, ix, iy) = 0.5 * (A[ix, iy] + A[ix, iy + 1])
Base.@propagate_inbounds av4(A, ix, iy) = 0.25 * (A[ix, iy] + A[ix + 1, iy] + A[ix, iy + 1] + A[ix + 1, iy + 1])

@kernel function update_stress!(Pr, Txx, Tyy, Txy, Vx, Vy, K, G, dt, dx, dy)
    ix, iy = @index(Global, NTuple)
    @inbounds if ix <= size(Pr, 1) && iy <= size(Pr, 2)
        exx = (Vx[ix + 1, iy] - Vx[ix, iy]) / dx
        eyy = (Vy[ix, iy + 1] - Vy[ix, iy]) / dy
        divV = exx + eyy
        Pr[ix, iy] -= divV * K[ix, iy] * dt
        Txx[ix, iy] += dt * 2.0 * G[ix, iy] * (exx - divV / 3.0)
        Tyy[ix, iy] += dt * 2.0 * G[ix, iy] * (eyy - divV / 3.0)
    end
    @inbounds if ix <= size(Txy, 1) && iy <= size(Txy, 2)
        exy = 0.5 * ((Vx[ix + 1, iy + 1] - Vx[ix + 1, iy]) / dy +
                     (Vy[ix + 1, iy + 1] - Vy[ix, iy + 1]) / dx)
        Txy[ix, iy] += dt * 2.0 * av4(G, ix, iy) * exy
    end
end

@kernel function update_velocity!(Vx, Vy, Pr, Txx, Tyy, Txy, rho, dt, dx, dy)
    ix, iy = @index(Global, NTuple)
    @inbounds if 1 < ix < size(Vx, 1) && 1 < iy < size(Vx, 2)
        Vx[ix, iy] += dt / avx(rho, ix - 1, iy) * (-(Pr[ix, iy] - Pr[ix - 1, iy]) / dx +
                                                   (Txx[ix, iy] - Txx[ix - 1, iy]) / dx +
                                                   (Txy[ix - 1, iy] - Txy[ix - 1, iy - 1]) / dy)
    end
    @inbounds if 1 < ix < size(Vy, 1) && 1 < iy < size(Vy, 2)
        Vy[ix, iy] += dt / avy(rho, ix, iy - 1) * (-(Pr[ix, iy] - Pr[ix, iy - 1]) / dy +
                                                   (Tyy[ix, iy] - Tyy[ix, iy - 1]) / dy +
                                                   (Txy[ix, iy - 1] - Txy[ix - 1, iy - 1]) / dx)
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
    K0     = 1.0
    G0     = 1.0
    rho0   = 1.0
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
    dt = dx / sqrt((K0 + 4 / 3 * G0) / rho0) / 2
    # array allocation
    Pr  = KA.zeros(backend, Float64, nx, ny)
    Txx = KA.zeros(backend, Float64, nx, ny)
    Tyy = KA.zeros(backend, Float64, nx, ny)
    Txy = KA.zeros(backend, Float64, nx - 1, ny - 1)
    Vx  = KA.zeros(backend, Float64, nx + 1, ny)
    Vy  = KA.zeros(backend, Float64, nx, ny + 1)
    G   = KA.zeros(backend, Float64, nx, ny)
    K   = KA.zeros(backend, Float64, nx, ny)
    rho = KA.zeros(backend, Float64, nx, ny)
    # init
    init_Pr!(backend, (32, 8), (nx, ny))(Pr, Lw, xc, yc)
    G   .= G0
    K   .= K0
    rho .= rho0
    KA.synchronize(backend)
    # write parameters
    write("out/dparams.dat", Lx, Ly, dx, dy)
    write("out/iparams.dat", nx, ny, nt, nsave)
    write("out/step_0.dat", Array(Pr))
    # action
    ttot = @elapsed begin
        for it in 1:nt
            update_stress!(backend, (32, 8), (nx, ny))(Pr, Txx, Tyy, Txy, Vx, Vy, K, G, dt, dx, dy)
            update_velocity!(backend, (32, 8), (nx + 1, ny + 1))(Vx, Vy, Pr, Txx, Tyy, Txy, rho, dt, dx, dy)
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
    size_rw = sizeof(Pr) + sizeof(Txx) + sizeof(Tyy) + sizeof(Txy) + sizeof(Vx) + sizeof(Vy)
    size_r  = sizeof(G) + sizeof(K) + sizeof(rho)
    GBs     = (2 * size_rw + 1 * size_r) / ttot / 1e9 * nt
    println("time = $ttot s, bandwidth = $GBs GB/s")
    return
end

main(CPU())
# main(CUDABackend())
# main(ROCBackend())