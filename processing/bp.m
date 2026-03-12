%% K-MC4 SAR Image Processor (Backprojection Algorithm)
%  Ce script charge les données brutes générées par le simulateur (index.m)
%  et forme une image SAR focalisée en utilisant l'algorithme Time-Domain Backprojection
%  adapté pour le FMCW (Frequency Domain Interpolation).

%clear; clc; close all;

%% 1. Chargement des Données et Configuration
fprintf('--- Chargement des données K-MC4 ---\n');
if ~isfile('output/KMC4_RawData.mat')
    error('Fichier KMC4_RawData.mat introuvable. Lancez d''abord index.m!');
end
load('output/KMC4_RawData.mat'); % Charge RawData_UP, RawData_DOWN, cfg, Pos_Radar

% Extraction des paramètres système depuis la config JSON chargée
c = 3e8;
fc = cfg.radar.fc;
lambda = c / fc;
B = cfg.radar.bandwidth;
T_sweep = cfg.modulation.sweep_time;
Fs = cfg.radar.adc_sample_rate;
K_slope = B / T_sweep;



[N_samples, N_pulses, N_channels] = size(RawData_UP);

fprintf('Paramètres chargés:\n- FC: %.2f GHz\n- Pente: %.2e Hz/s\n- Canaux: %d\n',...
    fc/1e9, K_slope, N_channels);

%% 1b. Visualisation des Données Brutes
figure('Name', 'Données Brutes', 'Position', [50, 50, 600, 400]);
imagesc(20*log10(abs(squeeze(RawData_DOWN(:, :, 1)))));
title('Amplitude des Données Brutes (Canal 1) [dB]');
xlabel('Index Impulsion');
ylabel('Index Échantillon');
colorbar; colormap('jet'); axis xy;

%% 2. Définition de la Grille de Sortie (Zone Image)
X_min = min(Pos_Radar(:,1)); X_max = max(Pos_Radar(:,1));
disp('X'); 
disp(X_min); 
disp(X_max); 
Y_min = 0;  Y_max = 10; 
Res_Grid = 0.05;            

Step_Range = c / (2 * B); 
Step_Azimuth = 0.05; 

x_vec = X_min : Res_Grid : X_max;
y_vec = Y_min : Res_Grid : Y_max;

fprintf('Pas de grille calculés : Range = %.3f m (Res Phys: %.2f m), Azimut = %.3f m\n', ...
    Res_Grid);
[GridX, GridY] = meshgrid(x_vec, y_vec);
[Ny, Nx] = size(GridX);
Image_Accumulator = zeros(Ny, Nx); 

fprintf('Grille image: %d x %d pixels (%.2f m x %.2f m)\n', Nx, Ny, (X_max-X_min), (Y_max-Y_min));

% 3. Pré-traitement : Compression en Distance (Range Compression)
fprintf('Exécution de la Range Compression (FFT)...\n');
Window = hanning(N_samples);
Win_Mat = repmat(Window, [1, N_pulses, N_channels]);

% Retrait de la moyenne (DC Offset) pour chaque canal et chaque pulse
RawData_UP = RawData_UP - mean(RawData_UP, 1);


N_FFT = 2^nextpow2(N_samples * 4); 
RC_Data_UP = fft(RawData_UP.* Win_Mat, N_FFT, 1);
 freq_axis = (0:N_FFT-1) * (Fs / N_FFT);


%% --- 3a. Visualisation de la Compression en Distance 2D (Range-Time Plot) ---
figure('Name', 'Range Compression 2D (RTI)', 'Position', [50, 450, 600, 400]);
range_axis = freq_axis * c / (2 * K_slope);
% Affichage de l'amplitude logarithmique des données compressées en distance
imagesc(1:N_pulses, range_axis, 20*log10(abs(RC_Data_UP(:, :, 1))));
% On limite l'affichage à une distance raisonnable (ex: 20m) pour y voir quelque chose
ylim([0, 20]);
title('Carte Range-Doppler / Profils de distance (Canal 1) [dB]');
xlabel('Index Impulsion (Azimut)');
ylabel('Distance [m]');
colorbar; colormap('jet'); axis xy; 

%% 3b. Fenêtrage Azimut
fprintf('Application du fenêtrage Azimut (Hanning)...\n');
Win_Az = hanning(N_pulses).'; 
RC_Data_UP = RC_Data_UP .* reshape(Win_Az, [1, N_pulses, 1]);

%% 4. Algorithme Backprojection FMCW (Le Cœur du Traitement)
fprintf('Lancement du Backprojection (Bistatique, Multi-Canal)...\n');
tic;

d_rx = cfg.antenna.rx_spacing_mm / 1000;
Rx_Offsets = zeros(N_channels, 3);
Rx_Offsets(:, 1) = linspace(-d_rx/2, d_rx/2, N_channels);

for m = 1:N_pulses
    if mod(m, 1000) == 0, fprintf('Traitement impulsion %d / %d\n', m, N_pulses); end
    
    P_Tx = Pos_Radar(m, :); 
    Dist_Tx_Grid = sqrt((GridX - P_Tx(1)).^2 + (GridY - P_Tx(2)).^2 + (0 - P_Tx(3)).^2);
    
    for ch = 1:N_channels
        P_Rx = P_Tx + Rx_Offsets(ch, :); 
        Dist_Rx_Grid = sqrt((GridX - P_Rx(1)).^2 + (GridY - P_Rx(2)).^2 + (0 - P_Rx(3)).^2);
        
        R_Bistatic = Dist_Tx_Grid + Dist_Rx_Grid;
        F_beat_Target = K_slope * (R_Bistatic / c);
        Bin_Idx = (F_beat_Target / (Fs / N_FFT)) + 1;
        
        Valid_Pixels = (Bin_Idx >= 1) & (Bin_Idx < N_FFT);
        Signal_Interp = zeros(size(GridX));
        Spectrum_Slice = RC_Data_UP(:, m, ch);
        
        idx_floor = floor(Bin_Idx(Valid_Pixels));
        idx_frac = Bin_Idx(Valid_Pixels) - idx_floor;
        
        val_low = Spectrum_Slice(idx_floor);
        val_high = Spectrum_Slice(idx_floor + 1);
        
        Signal_Interp(Valid_Pixels) = val_low.* (1 - idx_frac) + val_high.* idx_frac;
        Phase_Corr = exp(1j * 2 * pi * fc * (R_Bistatic(Valid_Pixels) / c));
        Image_Accumulator(Valid_Pixels) = Image_Accumulator(Valid_Pixels) + Signal_Interp(Valid_Pixels).* Phase_Corr;
    end
end
toc;

%% --- 4bis. INJECTION D'ERREUR (DÉSACTIVÉ) ---
fprintf('PGA appliqué sur l''image originale (pas de dégradation injectée)...\n');
Image_Degradee = Image_Accumulator;

%% --- 4ter. LANCEMENT DU PGA ---
fprintf('Lancement du PGA sur l''image dégradée...\n');
% On passe l'image dégradée au PGA. Vous pouvez ajuster la fenêtre initiale (ex: 200)
floor(Nx / 2) 
[Image_PGA, Phase_Error_Est, RMS] = pga_autofocus(Image_Degradee, 15);

%% 5. Visualisation et Analyse
figure("Name", 'Convergence RMS'); 
plot(RMS, 'b-o', 'LineWidth', 2);
title("Evolution du RMS en fonction de l'itération"); xlabel('Itérations'); ylabel('RMS Phase (rad)'); grid on;

figure('Name', 'Comparaison: Image Originale vs Image PGA', 'Position', [50, 200, 1200, 500]);

% A. Image Originale Formée
subplot(1,2,1);
Img_Mag = abs(Image_Accumulator);
Img_dB = 20*log10(Img_Mag / max(Img_Mag(:)));
imagesc(x_vec, y_vec, Img_dB);
colorbar; 
axis xy; axis equal; axis tight; colormap('jet'); caxis([-40 0]);
title('Image Originale (Avant PGA)');
xlabel('Azimut [m]'); ylabel('Distance [m]');

% B. Image Post-Autofocus (PGA)
subplot(1,2,2);
Img_Mag_PGA = abs(Image_PGA);
Img_dB_PGA = 20*log10(Img_Mag_PGA / max(Img_Mag_PGA(:)));
imagesc(x_vec, y_vec, Img_dB_PGA);
axis xy; axis equal; axis tight; colormap('jet'); caxis([-40 0]);
colorbar; 
title('Image Après PGA');
xlabel('Azimut [m]'); ylabel('Distance [m]');

% Superposition des cibles sur la dernière image
% hold on;
% for k = 1:length(cfg.scene.targets)
%     tgt = cfg.scene.targets(k);
%     plot(tgt.pos(1), tgt.pos(2), 'kp', 'MarkerSize', 12, 'MarkerFaceColor', 'w');
% end

% %% 5c. Visualisation du Tracking de Phase
% figure('Name', 'Estimation de Phase par PGA', 'Position', [200, 200, 800, 400]);
% % On centre la phase estimée
% Phase_Est_Centered = Phase_Error_Est - mean(Phase_Error_Est);
% plot(x_vec, Phase_Est_Centered, 'b-', 'LineWidth', 2);
% title("Estimation de l'erreur de phase par PGA");
% xlabel('Azimut [m]'); ylabel('Phase (rad)');
% grid on;

%% 6. Enregistrement des résultats
fprintf('Sauvegarde de l''image complexe dans output/out.mat...\n');
save('output/out.mat', 'Image_Accumulator', 'Image_Degradee', 'Image_PGA', 'x_vec', 'y_vec');
fprintf('Sauvegarde terminée.\n');