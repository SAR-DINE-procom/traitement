%% K-MC4 SAR Raw Data Simulator (Digital Twin) - Attitude Error Extension
clear; clc; close all;

%% 1. Chargement et Parsing de la Configuration
fprintf('--- Initialisation de la Simulation K-MC4 ---\n');
fid = fopen('conf/config3.json');
raw = fread(fid, inf);
str = char(raw');
fclose(fid);
cfg = jsondecode(str);

% Constantes Physiques
c = 3e8;

% Paramètres Dérivés
lambda_c = c / cfg.radar.fc;
K_slope = cfg.radar.bandwidth / cfg.modulation.sweep_time; 
N_samples = round(cfg.modulation.sweep_time * cfg.radar.adc_sample_rate);
dt = 1 / cfg.radar.adc_sample_rate;
time_axis = (0:N_samples-1) * dt;
pitch = deg2rad(cfg.platform.pitch_deg);

% Calcul de la trajectoire
dx = cfg.platform.velocity_mps / cfg.platform.prf;
N_pulses = round(cfg.platform.track_length_m / dx);
Pos_Radar = zeros(N_pulses, 3);
Pos_Radar(:,1) = linspace(0, cfg.platform.track_length_m, N_pulses) + cfg.platform.start_position(1);
Pos_Radar(:,2) = cfg.platform.start_position(2);
Pos_Radar(:,3) = cfg.platform.start_position(3);

%% --- NOUVEAU : Génération des erreurs Gauss-Markov ---
% On initialise à 0 par défaut si les champs n'existent pas dans le JSON
if ~isfield(cfg, 'motion'), cfg.motion.enabled = false; end

Errors_RPY = zeros(N_pulses, 3); % [Roll, Pitch, Yaw]
 
if cfg.motion.enabled
    disp('in')
    fprintf('Génération des erreurs d''attitude (Gauss-Markov)...\n');
    dt_slow = 1 / cfg.platform.prf;
    beta = exp(-dt_slow / cfg.motion.tau_corr);
    sigma_rad = deg2rad([cfg.motion.std_roll_deg, cfg.motion.std_pitch_deg, cfg.motion.std_yaw_deg]);
    
    for m = 2:N_pulses
        Errors_RPY(m,:) = beta * Errors_RPY(m-1,:) + sqrt(1 - beta^2) * (randn(1,3) .* sigma_rad);
    end

    % Errors_RPY = zeros(N_pulses, 3); % [Roll, Pitch, Yaw]
    % for m = 2:N_pulses
    %     Errors_RPY(m,1) = ((m - N_pulses / 2)* 0.02)^2;
    % end
end

%fprintf('Paramètres:\n- Freq: %.2f GHz\n- Pulses: %d\n- Erreurs Attitude: %s\n',...
%    cfg.radar.fc/1e9, N_pulses, regexprep(num2str(cfg.motion.enabled),'1','OUI','0','NON'));

%% 2. Initialisation des Matrices de Données
RawData_UP   = complex(zeros(N_samples, N_pulses, 2)); 
RawData_DOWN = complex(zeros(N_samples, N_pulses, 2));

%% 3. Boucle de Simulation (Slow Time)
fprintf('Lancement de la simulation (Stop-and-Go)...\n');
tic;

d_rx = cfg.antenna.rx_spacing_mm / 1000;
Offset_Rx1_Nominal = [-d_rx/2, 0, 0];
Offset_Rx2_Nominal = [ d_rx/2, 0, 0];

for m = 1:N_pulses
    P_Tx_Global = Pos_Radar(m, :);
    
    % --- NOUVEAU : Calcul de la rotation pour cette impulsion ---
    % Correction des axes : Le radar regarde vers l'axe Y.
    % - Pour s'incliner vers le haut/bas (tangage/pitch nominal pour regarder le sol), 
    %   il faut tourner autour de l'axe X (orthogonal à la ligne de visée Y).
    r = Errors_RPY(m,1); p = Errors_RPY(m,2); y = Errors_RPY(m,3);
    Rx = [1 0 0; 0 cos(p + pitch) -sin(p + pitch); 0 sin(p + pitch) cos(p + pitch)];
    Ry = [cos(r) 0 sin(r); 0 1 0; -sin(r) 0 cos(r)];
    Rz = [cos(y) -sin(y) 0; sin(y) cos(y) 0; 0 0 1];
    R_total = Rz * Ry * Rx; % Matrice de rotation globale

    % Rotation des offsets d'antenne (pour l'interférométrie)
    Off1 = (R_total * Offset_Rx1_Nominal')';
    Off2 = (R_total * Offset_Rx2_Nominal')';
 

    Signal_Acc_UP_Rx1 = zeros(1, N_samples); Signal_Acc_UP_Rx2 = zeros(1, N_samples);
    Signal_Acc_DOWN_Rx1 = zeros(1, N_samples); Signal_Acc_DOWN_Rx2 = zeros(1, N_samples);
    
    for k = 1:length(cfg.scene.targets)
        P_Tgt = cfg.scene.targets(k).pos';
        Vec_R = P_Tgt - P_Tx_Global;
        Dist_Tx = norm(Vec_R);
        
        % --- NOUVEAU : Projection dans le repère local tourné ---
        % Nécessaire pour le gain d'antenne (le radar "voit" la cible sous un autre angle)
        Vec_R_Local = R_total' * Vec_R'; 
        
        angle_az = atan2(Vec_R_Local(1), Vec_R_Local(2)); 
        angle_el = atan2(Vec_R_Local(3), sqrt(Vec_R_Local(1)^2 + Vec_R_Local(2)^2));
        
        % Gain et Amplitude
        Gain_Az = exp(-4*log(2) * (angle_az / deg2rad(cfg.antenna.beamwidth_azimuth_deg))^2);
        Gain_El = exp(-4*log(2) * (angle_el / deg2rad(cfg.antenna.beamwidth_elevation_deg))^2);
        Gain_Total_Lin = 10^(cfg.antenna.gain_dbi/10) * Gain_Az * Gain_El;
        Amp_k = sqrt(cfg.scene.targets(k).rcs) * Gain_Total_Lin / (Dist_Tx^2);
        
        % Distances Rx avec Offsets tournés
        Dist_Rx1 = Dist_Tx + dot(Vec_R/Dist_Tx, Off1);
        Dist_Rx2 = Dist_Tx + dot(Vec_R/Dist_Tx, Off2);
        
        tau_1 = (Dist_Tx + Dist_Rx1) / c;
        tau_2 = (Dist_Tx + Dist_Rx2) / c;
        
        Phi_fixe_1 = 2 * pi * cfg.radar.fc * tau_1;
        Phi_fixe_2 = 2 * pi * cfg.radar.fc * tau_2;
        fb_1 = K_slope * tau_1;
        fb_2 = K_slope * tau_2;
        
        % Synthèse
        Sig_Up_1 = Amp_k * exp(1j * (2*pi*fb_1*time_axis + Phi_fixe_1));
        Sig_Up_2 = Amp_k * exp(1j * (2*pi*fb_2*time_axis + Phi_fixe_2));
        Sig_Down_1 = Amp_k * exp(1j * (2*pi*(-fb_1)*time_axis + Phi_fixe_1));
        Sig_Down_2 = Amp_k * exp(1j * (2*pi*(-fb_2)*time_axis + Phi_fixe_2));
        
        Signal_Acc_UP_Rx1 = Signal_Acc_UP_Rx1 + Sig_Up_1;
        Signal_Acc_UP_Rx2 = Signal_Acc_UP_Rx2 + Sig_Up_2;
        Signal_Acc_DOWN_Rx1 = Signal_Acc_DOWN_Rx1 + Sig_Down_1;
        Signal_Acc_DOWN_Rx2 = Signal_Acc_DOWN_Rx2 + Sig_Down_2;
    end
    
    RawData_UP(:, m, 1) = Signal_Acc_UP_Rx1;
    RawData_UP(:, m, 2) = Signal_Acc_UP_Rx2;
    RawData_DOWN(:, m, 1) = Signal_Acc_DOWN_Rx1;
    RawData_DOWN(:, m, 2) = Signal_Acc_DOWN_Rx2;
end
toc;

%% --- NOUVEAU : Ajout de la Réverbération de Sol (Modèle Statistique) ---
% On simule le sol comme une infinité de diffuseurs aléatoires (Speckle)
% Cela revient à ajouter un bruit Gaussien Complexe aux données brutes.

% 1. Paramétrage de la puissance du sol (Clutter)
% Ajustez ce facteur pour rendre le sol plus ou moins brillant par rapport aux cibles
% Une valeur de 1e-4 est souvent un bon point de départ par rapport à des cibles ponctuelles
Clutter_Power = 0; 

if isfield(cfg, 'scene') && isfield(cfg.scene, 'clutter_power')
    Clutter_Power = cfg.scene.clutter_power;
end

fprintf('Génération du Clutter de sol (Modèle Rayleigh, Puissance: %.1e)...\n', Clutter_Power);

% 2. Génération du signal de sol (Bruit Gaussien Complexe)
% Note : Pour de l'InSAR (2 antennes), le sol est "vu" presque identiquement par les deux antennes.
% On génère donc une "Vérité Terrain" de sol unique.
Ground_Reflectivity = sqrt(Clutter_Power/2) * (randn(N_samples, N_pulses) + 1j * randn(N_samples, N_pulses));

% 3. Ajout aux données brutes (Superposition linéaire)
% On l'ajoute aux deux canaux Rx. 
% Note: Dans un modèle plus avancé, on ajouterait un déphasage géométrique entre Rx1 et Rx2.
RawData_UP(:,:,1)   = RawData_UP(:,:,1)   + Ground_Reflectivity;
RawData_UP(:,:,2)   = RawData_UP(:,:,2)   + Ground_Reflectivity;
RawData_DOWN(:,:,1) = RawData_DOWN(:,:,1) + Ground_Reflectivity;
RawData_DOWN(:,:,2) = RawData_DOWN(:,:,2) + Ground_Reflectivity;

fprintf('Clutter ajouté avec succès.\n');

%% 4. Visualisation
figure('Name', 'K-MC4 SAR Simulator - Analyse Globale', 'Color', 'w', 'Position', [100, 100, 1200, 800]);

% 1. Erreurs temporelles (Attitude)
subplot(2,2,1);
t_slow = (0:N_pulses-1) / cfg.platform.prf;
plot(t_slow, rad2deg(Errors_RPY), 'LineWidth', 1.2);
grid on; title('Erreurs d''Attitude (Gauss-Markov)');
xlabel('Temps (s)'); ylabel('Angle (deg)');
legend('Roulis', 'Tangage', 'Lacet');

% 2. Données Brutes (Partie Réelle) - Historique de phase
subplot(2,2,2);
imagesc(1:N_pulses, time_axis*1000, real(RawData_UP(:,:,1)));
colormap(gca, 'gray'); colorbar;
title('Données Brutes (Partie Réelle) - Franges SAR');
xlabel('Numéro Impulsion'); ylabel('Temps rapide (ms)');

% --- Limites statiques en dB pour l'affichage ---
c_min_db = -40; % À ajuster selon la puissance de vos cibles
c_max_db = 50;  % À ajuster selon la puissance de vos cibles

% 3. RTI (Range-Time Intensity)
subplot(2,2,3);
f = (cfg.radar.adc_sample_rate/1000) * (0:(N_samples/2))/N_samples;
fft_raw = fft(RawData_UP(:,:,1));
imagesc(1:N_pulses, f, db(abs(fft_raw(1:length(f), :))));
clim([c_min_db c_max_db]); % Fixe la dynamique des couleurs (utiliser caxis() sur les vieux MATLAB)
ylim([0 max(f)/2]); colormap(gca, 'jet'); colorbar;
title('RTI (Spectre Fast-Time vs Slow-Time)');
xlabel('Numéro Impulsion'); ylabel('Fréquence (kHz)');

% 4. Profil de Fréquence de Battement (Moyenné)
subplot(2,2,4);
mean_fft_profile = mean(abs(fft_raw(1:length(f), :)), 2);
plot(f, db(mean_fft_profile), 'LineWidth', 1.5, 'Color', '#0072BD');
ylim([c_min_db c_max_db]); % Fixe la dynamique de l'axe vertical
grid on;
title('Profil de Fréquence de Battement (Moyenné)');
xlabel('Fréquence (kHz)'); ylabel('Amplitude Moyenne (dB)');
xlim([0 max(f)/2]);

save('output/KMC4_RawData.mat', 'RawData_UP', 'RawData_DOWN', 'cfg', 'Errors_RPY', 'Pos_Radar');
fprintf('Sauvegarde terminée (avec Erreurs et Trajectoire).\n');