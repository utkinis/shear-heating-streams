using CairoMakie

nx, ny, nt, nsave = reinterpret(Int    , read("out/iparams.dat"))
Lx, Ly, dx, dy    = reinterpret(Float64, read("out/dparams.dat"))

fig = Figure(resolution=(1200,1000),fontsize=32)
ax  = (
    P_ini = Axis(fig[1,1][1,1], aspect=DataAspect(), title="p₀"),
    T_ini = Axis(fig[1,2][1,1], aspect=DataAspect(), title="T₀"),
    P     = Axis(fig[2,1][1,1], aspect=DataAspect(), title="p"),
    T     = Axis(fig[2,2][1,1], aspect=DataAspect(), title="T"),
)

title = Label(fig[0, :], "0")

P = Array{Float64}(undef, nx, ny)
T = Array{Float64}(undef, nx, ny)

plt = (
    P_ini = heatmap!(ax.P_ini, P, colormap=:turbo),
    T_ini = heatmap!(ax.T_ini, T, colormap=:turbo),
    P     = heatmap!(ax.P    , P, colormap=:turbo),
    T     = heatmap!(ax.T    , T, colormap=:turbo, colorrange=(0,1)),
)

Colorbar(fig[1,1][1,2], plt.P_ini)
Colorbar(fig[1,2][1,2], plt.T_ini)
Colorbar(fig[2,1][1,2], plt.P)
Colorbar(fig[2,2][1,2], plt.T)

for it in 0:nsave:nt
    open("out/step_$it.dat","r") do io
        read!(io, P)
        read!(io, T)
    end
    if it == 0
        plt.P_ini[1] = P
        plt.T_ini[1] = T
    end
    plt.P[1]  = P
    plt.T[1]  = T
    title.text = "$it"
    display(fig)
end