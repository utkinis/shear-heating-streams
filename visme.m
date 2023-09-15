clear; figure(1); clf; colormap turbo;

fid  = fopen("iparams.dat", "r");
iparams = fread(fid, 2, "int64");
fclose(fid);

fid  = fopen("dparams.dat", "r");
dparams = fread(fid, 4, "double");
fclose(fid);

nx = iparams(1)
ny = iparams(2)

Lx = dparams(1)
Ly = dparams(2)
dx = dparams(3)
dy = dparams(4)

fid    = fopen("Pr.dat", "r");
Pr_ini = fread(fid, [nx ny], "double");
Pr     = fread(fid, [nx ny], "double");
fclose(fid);

subplot(121); imagesc(Pr_ini); axis image; colorbar
subplot(122); imagesc(Pr)    ; axis image; colorbar
