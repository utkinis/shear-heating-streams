clear; figure(1); clf; colormap turbo;

fid  = fopen("iparams.dat", "r");
iparams = num2cell(fread(fid, 4, "int64"));
[nx, ny, nt, nsave] = deal(iparams{:})
fclose(fid);

fid  = fopen("dparams.dat", "r");
dparams = num2cell(fread(fid, 4, "double"));
[Lx, Ly, dx, dy] = deal(dparams{:})
fclose(fid);

tiledlayout(2,2, "TileSpacing", "tight", "Padding", "tight")

for it = 0:nsave:nt
    fid    = fopen(['step_' num2str(it) '.dat'], "r");
    Pr     = fread(fid, [nx ny], "double");
    T      = fread(fid, [nx ny], "double");
    fclose(fid);

    sgtitle(it)

    if it == 0
        nexttile(1); imagesc(Pr); axis image; colorbar             ; title("p_0")
        nexttile(2); imagesc(T) ; axis image; colorbar; clim([0 1]); title("T_0")
    end
    
    nexttile(3); imagesc(Pr); axis image; colorbar             ; title("p")
    nexttile(4); imagesc(T) ; axis image; colorbar; clim([0 1]); title("T")

    drawnow
end