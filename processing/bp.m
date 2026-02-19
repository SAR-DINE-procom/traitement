%% K-MC4 SAR Image Processor (Backprojection Algorithm)
%  Ce script charge les données brutes générées par le simulateur (index.m)
%  et forme une image SAR focalisée en utilisant l'algorithme Time-Domain Backprojection
%  adapté pour le FMCW (Frequency Domain Interpolation).

% clear; clc; close all;

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

%% 2. Définition de la Grille de Sortie (Zone Image)
% On définit une zone centrée sur les cibles définies dans le JSON
% Vous pouvez ajuster ces limites manuellement selon votre scène
X_min = 0.0; X_max = cfg.platform.track_length_m;  % Axe Azimut (Cross-range) - Centré sur le target à 1.0m
Y_min = 0.0;  Y_max = 10.0; % Axe Distance (Range) - Zoom sur 5-20m
Res_Grid = 0.1;            % Résolution de la grille (5 cm) - DEPRECATED

% Calcul des Pas de Grille (Grid Sampling Steps) basés sur la physique
% 1. Axe Distance : Dépend de la Bande Passante (B)
% Résolution théorique = c / 2B. On échantillonne plus fin (ex: /8) pour le confort visuel.
Step_Range = c / (2 * B); 

% 2. Axe Azimut : Dépend de la tailel de l'antenne
% KCM4 =~ 10cm
Step_Azimuth = 0.5; 

%x_vec = X_min : Step_Azimuth : X_max;
%y_vec = Y_min : Step_Range : Y_max;

x_vec = X_min : Res_Grid : X_max;
y_vec = Y_min : Res_Grid : Y_max;

fprintf('Pas de grille calculés : Range = %.3f m (Res Phys: %.2f m), Azimut = %.3f m\n', ...
    Res_Grid);
[GridX, GridY] = meshgrid(x_vec, y_vec);
[Ny, Nx] = size(GridX);
Image_Accumulator = zeros(Ny, Nx); % Image complexe finale

fprintf('Grille image: %d x %d pixels (%.2f m x %.2f m)\n', Nx, Ny, (X_max-X_min), (Y_max-Y_min));

%% 3. Pré-traitement : Compression en Distance (Range Compression)
% En FMCW, la "compression" est une simple FFT sur l'axe fast-time.
% On applique un fenêtrage pour réduire les lobes secondaires (Side Lobes).

fprintf('Exécution de la Range Compression (FFT)...\n');
Window = hanning(N_samples);
% On duplique la fenêtre pour l'appliquer à toutes les impulsions et canaux
Win_Mat = repmat(Window, [1, N_pulses, N_channels]);

% FFT sur la dimension 1 (Fast Time)
% On ajoute du Zero-Padding (ex: 4x) pour améliorer la précision de l'interpolation plus tard
N_FFT = 2^nextpow2(N_samples * 4); 
RC_Data_UP = fft(RawData_UP.* Win_Mat, N_FFT, 1);

% Création du vecteur fréquence associé aux bins de la FFT
freq_axis = (0:N_FFT-1) * (Fs / N_FFT);

%% 3b. Fenêtrage Azimut (Optionnel mais recommandé)
% Applique une fenêtre (ex: Hanning) sur l'axe Slow-Time pour réduire 
% les lobes secondaires latéraux causés par la fin abrupte de l'acquisition.
fprintf('Application du fenêtrage Azimut (Hanning)...\n');
Win_Az = hanning(N_pulses).'; % Vecteur ligne [1 x N_pulses]
% On applique la fenêtre à RC_Data_UP [N_FFT, N_pulses, N_channels]
RC_Data_UP = RC_Data_UP .* reshape(Win_Az, [1, N_pulses, 1]);

%% 4. Algorithme Backprojection FMCW (Le Cœur du Traitement)
fprintf('Lancement du Backprojection (Bistatique, Multi-Canal)...\n');
tic;

% Offsets des antennes réceptrices (Interférométrie / Bistaticité)
% Tx est supposé à l'origine dans le repère capteur local
d_rx = cfg.antenna.rx_spacing_mm / 1000;
Rx_Offsets = zeros(N_channels, 3);
% Le simulateur génère: Offset_Rx1 = -d/2, Offset_Rx2 = +d/2
Rx_Offsets(:, 1) = linspace(-d_rx/2, d_rx/2, N_channels);

% Pour optimiser la vitesse, on vectorise sur les pixels mais on boucle sur les impulsions
for m = 1:N_pulses
    if mod(m, 1000) == 0, fprintf('Traitement impulsion %d / %d\n', m, N_pulses); end
    
    % Position globale de l'antenne TX à l'instant m
    P_Tx = Pos_Radar(m, :); 
    
    % --- Calculs vectorisés pour toute la grille image ---
    % Distances de chaque pixel (x,y,0) au Tx
    % Z_pixel est supposé à 0 (Sol)
    Dist_Tx_Grid = sqrt((GridX - P_Tx(1)).^2 + (GridY - P_Tx(2)).^2 + (0 - P_Tx(3)).^2);
    
    % Boucle sur les canaux RX (MIMO / Monopulse processing)
    for ch = 1:N_channels
        % Position globale de l'antenne RX 'ch' à l'instant m
        % On applique simplement l'offset local à la position globale du radar
        P_Rx = P_Tx + Rx_Offsets(ch, :); 
        
        % Distances de chaque pixel au Rx courant
        Dist_Rx_Grid = sqrt((GridX - P_Rx(1)).^2 + (GridY - P_Rx(2)).^2 + (0 - P_Rx(3)).^2);
        
        % Distance Bistatique Totale (Aller-Retour)
        R_Bistatic = Dist_Tx_Grid + Dist_Rx_Grid;
        
        % --- A. Mapping Distance -> Fréquence (Spécifique FMCW) ---
        % f_b = K * tau = K * (R_total / c)
        F_beat_Target = K_slope * (R_Bistatic / c);
        
        % Conversion Fréquence -> Index FFT (Bin)
        % idx = (freq / df) + 1
        Bin_Idx = (F_beat_Target / (Fs / N_FFT)) + 1;
        
        % --- B. Sélection et Interpolation ---
        % On ne garde que les pixels valides (dans la bande passante numérisée)
        Valid_Pixels = (Bin_Idx >= 1) & (Bin_Idx < N_FFT);
        
        % Extraction des valeurs complexes par interpolation linéaire
        % interp1 est lent sur des matrices 2D, on utilise l'indexation linéaire pour la vitesse
        Signal_Interp = zeros(size(GridX));
        
        % Récupération du profil de distance pour ce pulse et ce canal
        Spectrum_Slice = RC_Data_UP(:, m, ch);
        
        % Interpolation manuelle rapide pour les indices valides
        idx_floor = floor(Bin_Idx(Valid_Pixels));
        idx_frac = Bin_Idx(Valid_Pixels) - idx_floor;
        
        val_low = Spectrum_Slice(idx_floor);
        val_high = Spectrum_Slice(idx_floor + 1);
        
        Signal_Interp(Valid_Pixels) = val_low.* (1 - idx_frac) + val_high.* idx_frac;
        
        % --- C. Compensation de Phase (Matched Filter) ---
        % C'est l'étape CRITIQUE. Le simulateur a injecté exp(+j * 2*pi*fc * tau).
        % Le Backprojection doit multiplier par exp(-j * 2*pi*fc * tau) pour annuler
        % la phase de propagation et focaliser l'énergie.
        
        Phase_Corr = exp(-1j * 2 * pi * fc * (R_Bistatic(Valid_Pixels) / c));
        
        % --- D. Compensation RVP (Residual Video Phase) ---
        % Le terme RVP est exp(-j * pi * K * tau^2). 
        % Pour le corriger, on multiplie par le conjugué: exp(+j * pi * K * tau^2).
        tau_grid = R_Bistatic(Valid_Pixels) / c;
        RVP_Corr = exp(1j * pi * K_slope * tau_grid.^2);
        
        % Accumulation Cohérente dans l'image globale
        % On somme : Signal_Interpolé * Correction_Phase * Correction_RVP
        %Image_Accumulator(Valid_Pixels) = Image_Accumulator(Valid_Pixels) +...
            %Signal_Interp(Valid_Pixels).* Phase_Corr.* RVP_Corr;
        Image_Accumulator(Valid_Pixels) = Image_Accumulator(Valid_Pixels) + Signal_Interp(Valid_Pixels).* Phase_Corr;
    end
end
toc;
[Image_PGA, Phase_Error] = pga_autofocus(Image_Accumulator, 5);
%% 5. Visualisation et Analyse
figure('Name', 'K-MC4 SAR Image');

% Affichage de l'image (Magnitude Logarithmique)
% On normalise par rapport au max pour avoir des dB
Img_Mag = abs(Image_Accumulator);
Img_dB = 20*log10(Img_Mag / max(Img_Mag(:)));

imagesc(x_vec, y_vec, Img_dB);
axis xy; axis equal; axis tight;
colormap('jet');
caxis([-100 0]); % Dynamique de 40 dB pour voir les lobes secondaires
colorbar;
title('Image SAR Reconstruite (K-MC4 Backprojection)');
xlabel('Azimut / Cross-Range [m]');
ylabel('Distance / Range [m]');
grid on;

% Superposition des vraies cibles (Ground Truth) depuis le fichier de config
hold on;
for k = 1:length(cfg.scene.targets)
    tgt = cfg.scene.targets(k);
    plot(tgt.pos(1), tgt.pos(2), 'kp', 'MarkerSize', 12, 'MarkerFaceColor', 'w');
    text(tgt.pos(1)+0.1, tgt.pos(2), sprintf('Tgt %d', tgt.id), 'Color', 'w', 'FontWeight', 'bold');
end
legend('Ground Truth');

fprintf('Image affichée. Les marqueurs indiquent la position réelle des cibles.\n');

%% 5b. Visualisation Autofocus
figure('Name', 'K-MC4 SAR Image Post-Autofocus (PGA)', 'Position', [150, 150, 800, 600]);

Img_Mag_PGA = abs(Image_PGA);
Img_dB_PGA = 20*log10(Img_Mag_PGA / max(Img_Mag_PGA(:)));

imagesc(x_vec, y_vec, Img_dB_PGA);
axis xy; axis equal;
colormap('jet');
caxis([-100 0]); 
colorbar;
title('Image SAR Après Autofocus (PGA)');
xlabel('Azimut / Cross-Range [m]');
ylabel('Distance / Range [m]');
grid on;

% Superposition des cibles
hold on;
for k = 1:length(cfg.scene.targets)
    tgt = cfg.scene.targets(k);
    plot(tgt.pos(1), tgt.pos(2), 'kp', 'MarkerSize', 12, 'MarkerFaceColor', 'w');
    text(tgt.pos(1)+0.1, tgt.pos(2), sprintf('Tgt %d', tgt.id), 'Color', 'w', 'FontWeight', 'bold');
end
legend('Ground Truth');

%% 5c. Visualisation Erreur de Phase et Théorie
figure('Name', 'Estimation vs Théorie (Phase Error)', 'Position', [200, 200, 800, 500]);
hold on;
% 1. Erreur PGA (Alignée approximativement)
plot(x_vec, rad2deg(Phase_Error), 'b-', 'LineWidth', 1.5, 'DisplayName', 'Estimée (PGA, img)');

% 2. Erreur Théorique (Calculée à partir des mouvements)
if exist('Errors_RPY', 'var') && ~isempty(Errors_RPY)
    % Calcul de l'erreur de phase induite pour le canal 1 sur la cible 1
    % On simule ce que le mouvement induit comme changement de distance RX
    
    phi_theory = zeros(1, N_pulses);
    Tgt_Pos = cfg.scene.targets(1).pos; % Utiliser la première cible
    % Assurer que Tgt_Pos est un vecteur ligne (1x3) comme Pos_Radar
    if iscolumn(Tgt_Pos), Tgt_Pos = Tgt_Pos'; end

    % Offsets nominaux (doivent correspondre à la simulation)
    drx = cfg.antenna.rx_spacing_mm / 1000;
    Off0 = [-drx/2, 0, 0]; 
    
    fprintf('Calcul erreur théorique (Target 1, Canal 1)...\n');
    for m = 1:N_pulses
        % Vecteur vue
        P_Radar = Pos_Radar(m, :);
        LOS = Tgt_Pos - P_Radar;
        u_LOS = LOS / norm(LOS);
        
        % Rotation
        r = Errors_RPY(m,1); p = Errors_RPY(m,2); y = Errors_RPY(m,3);
        Rx = [1 0 0; 0 cos(r) -sin(r); 0 sin(r) cos(r)];
        Ry = [cos(p) 0 sin(p); 0 1 0; -sin(p) 0 cos(p)];
        Rz = [cos(y) -sin(y) 0; sin(y) cos(y) 0; 0 0 1];
        R_tot = Rz * Ry * Rx;
        
        % Offset Réel
        Off_Real = (R_tot * Off0')';
        
        % Erreur de distance RX (Projection sur LOS)
        d_err = dot(u_LOS, Off_Real - Off0);
        
        % Erreur de phase (2pi / lambda * d_err)
        % Note: Bistatique. Tx ne bouge pas (CG). Seul Rx bouge.
        phi_theory(m) = (2*pi/lambda) * d_err;
    end
    
    % On plot en fonction de la position X du radar pour aligner avec l'image
    plot(Pos_Radar(:,1), rad2deg(phi_theory), 'r--', 'LineWidth', 1.2, 'DisplayName', 'Théorique (Motion Ch1)');
    title('Comparaison Erreur Phase: Estimée vs Théorique (Motion)');
else
    title('Erreur de Phase Estimée par PGA');
end

xlabel('Position Azimut (m)');
ylabel('Erreur de Phase (Degrés)');
legend('Location', 'best');
grid on;
xlim([X_min X_max]);

%% 5d. Visualisation de la Phase de l'image
figure('Name', 'Phase de l''image SAR', 'Position', [250, 250, 800, 600]);
imagesc(x_vec, y_vec, angle(Image_Accumulator));
axis xy; axis equal; axis tight;
colormap('hsv'); % Colormap cyclique adaptée pour la phase (-pi à pi)
colorbar;
title('Phase de l''image SAR Reconstruite');
xlabel('Azimut / Cross-Range [m]');
ylabel('Distance / Range [m]');
grid on;

%% 6. Enregistrement des résultats
fprintf('Sauvegarde de l''image complexe dans output/out.mat...\n');
save('output/out.mat', 'Image_Accumulator', 'Image_PGA', 'x_vec', 'y_vec');
fprintf('Sauvegarde terminée.\n');
