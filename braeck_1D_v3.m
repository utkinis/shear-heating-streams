clear,figure(1),clf,colormap(jet)
% dimensionally independent physics
h     = 1.0; % m
sig0  = 1.0; % Pa
taur  = 1.0; % s
E_R   = 1.0; % K
% non-dimensional parameters
npow        = 3;
h_L         = 5e-2;
T0_E_R      = 2e-2;
Tbg_E_R     = 5e-4;
sig0_sigc   = 5e0;
taur_taud   = 5e-2;
% definitions
mu0_mubg    = exp(1 / T0_E_R - 1 / Tbg_E_R);
Deltap      = h_L + (1.0 - h_L) * mu0_mubg;
% dimensionally dependent physics
L           = h / h_L;
sigc        = sig0 / sig0_sigc;
taud        = taur / taur_taud;
%ttot        = min(0.6*taur, 0.1*taud);
T0          = T0_E_R * E_R;
Tbg         = Tbg_E_R * E_R;
Xi          = h^2 / taud;
A_C   = exp(E_R / T0) * sig0^(1 - npow) * T0^2 / (taur * sigc^2 * E_R);
AG    = exp(E_R / T0) * sig0^(1 - npow) / (2 * taur * Deltap);
Tmaxa = T0_E_R + sig0^2 * L / (2 * AG / A_C * h) / E_R;
% numerics
nx     = 2000;
niter  = 5;
nvis   = 100;
epstol = 1e-10;
% preprocessing
dx      = L / nx;
xv      = linspace(-L / 2, L / 2, nx + 1);
xc      = avx(xv);
dt_diff = dx^2 / Xi / 10.1;
dt      = dt_diff;
%nt      = ceil(ttot / dt);
% fields
T       = zeros(1,nx) + Tbg;
T_old   = zeros(1,nx);
qTx     = zeros(1,nx + 1);
Vx      = zeros(1,nx + 1);
Ux      = zeros(1,nx + 1);
% initialisation
T(abs(xc)<= h/2) = T0;
sig      = sig0;
% temporal evolution
time_evo = 0;
Tmax_evo = max(T)/E_R;
sig_evo  = sig0;
Vmax_evo = 0;
% time loop
it       = 1;  
tcur     = 0.0;
while  Tmax_evo(end) > max(Tmax_evo)/5  
    sig_old = sig;
    T_old = T;
    for iter = 1:niter
        dsig_dt = -AG/L*sig^(npow-1)*sum(exp(-E_R./T))*dx;
        dt      = min([1e-3*sig0/abs(sig*dsig_dt),dt_diff,1e-3*taur]);
        sig           = sig_old/(1 - dt*dsig_dt);
        qTx(2:end-1)  = -Xi*diff(T)./ dx;
        T             = T_old  + dt*(-diff(qTx)./ dx  ...
            +                         A_C.*sig^(npow + 1).*exp(-E_R./T));
    end
    Vx(2:end) = cumsum((sig^npow.*exp(-E_R./T)+(sig - sig_old)/dt/AG).*dx);    
    Ux    = Ux   + Vx*dt;
    tcur  = tcur +    dt;
    it    = it   +     1;
    Tmax_evo(it) = max(T)/E_R;
    sig_evo(it)  = sig;
    Vmax_evo(it) = max(abs(Vx));
    if mod(it,nvis)==0 || Tmax_evo(end) <= max(Tmax_evo)/5 
        time = (1:it)*dt;
        subplot(221)%,plot(xv,Ux),title('Ux')
        semilogy(time,Vmax_evo),title('Vmax(t)')
        subplot(222),plot(xc,T),yline(Tmaxa,'--'),title('T(x)'),xlim([-h h]/2)
        subplot(223),plot(time,Tmax_evo),title('Tmax(t)')
        subplot(224),plot(time,sig_evo),title('\sigma(t)')
        drawnow
    end
end
Tmax_nd = (max(Tmax_evo) - T0_E_R) / (Tmaxa - T0_E_R)
function a = avx(a)
a = 0.5 .* (a(1:end-1)  + a(2:end));
end