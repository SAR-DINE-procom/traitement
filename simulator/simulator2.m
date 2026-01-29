%% K-MC4 SAR Raw Data Simulator (Digital Twin)
%  Génère des données brutes FMCW (Signal de Battement) basées sur la config JSON.
%  Supporte: Modulation Triangulaire, Diagramme d'Antenne, Double Canal RX.

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
K_slope = cfg.radar.bandwidth / cfg.modulation.sweep_time; % Pente (Hz/s)
N_samples = round(cfg.modulation.sweep_time * cfg.radar.adc_sample_rate);
dt = 1 / cfg.radar.adc_sample_rate;
time_axis = (0:N_samples-1) * dt;

% Calcul de la trajectoire (Stop-and-Go)
% PRF détermine le pas spatial : dx = v / PRF
dx = cfg.platform.velocity_mps / cfg.platform.prf;
N_pulses = round(cfg.platform.track_length_m / dx);
Pos_Radar = zeros(N_pulses, 3);
Pos_Radar(:,1) = linspace(0, cfg.platform.track_length_m, N_pulses) + cfg.platform.start_position(1);
Pos_Radar(:,2) = cfg.platform.start_position(2);
Pos_Radar(:,3) = cfg.platform.start_position(3);

fprintf('Paramètres:\n- Freq: %.2f GHz\n- BW: %.0f MHz\n- Pulses: %d\n- Samples/Chirp: %d\n',...
    cfg.radar.fc/1e9, cfg.radar.bandwidth/1e6, N_pulses, N_samples);

%% 2. Initialisation des Matrices de Données (Memory Pre-allocation)
% Structure:
% On sépare Up et Down pour faciliter le traitement ultérieur
RawData_UP   = complex(zeros(N_samples, N_pulses, 2)); % Rx1 et Rx2
RawData_DOWN = complex(zeros(N_samples, N_pulses, 2));

%% 3. Boucle de Simulation (Slow Time)
fprintf('Lancement de la simulation (Stop-and-Go)...\n');
tic;

% Définition des positions relatives des antennes Rx par rapport au Tx (au centre)
% K-MC4: Tx au centre, Rx1 et Rx2 espacés. 
% Simplification: Tx à 0, Rx1 à -d/2, Rx2 à +d/2 (selon l'axe X azimut)
d_rx = cfg.antenna.rx_spacing_mm / 1000;
Offset_Rx1 = [-d_rx/2, 0, 0];
Offset_Rx2 = [ d_rx/2, 0, 0];

for m = 1:N_pulses
    % Position globale du centre de phase Tx à l'impulsion m
    P_Tx_Global = Pos_Radar(m, :);
    
    % Accumulateurs pour ce chirp (somme complexe des contributions cibles)
    Signal_Acc_UP_Rx1   = zeros(1, N_samples);
    Signal_Acc_UP_Rx2   = zeros(1, N_samples);
    Signal_Acc_DOWN_Rx1 = zeros(1, N_samples);
    Signal_Acc_DOWN_Rx2 = zeros(1, N_samples);
    
    for k = 1:length(cfg.scene.targets)
        Tgt = cfg.scene.targets(k);
        P_Tgt = Tgt.pos';
        
        % --- A. Géométrie et Diagramme d'Antenne ---
        
        % Vecteur Radar -> Cible (Référentiel global)
        Vec_R = P_Tgt - P_Tx_Global;
        Dist_Tx = norm(Vec_R);
        
        % Calcul des angles par rapport à la ligne de visée (Boresight = axe Y)
        % Azimut (Plan XY)
        angle_az = atan2(Vec_R(1), Vec_R(2)); 
        % Elevation (Plan YZ)
        angle_el = atan2(Vec_R(3), sqrt(Vec_R(1)^2 + Vec_R(2)^2));
        
        % Gain Antenne (Modèle Gaussien Simplifié)
        % G(theta) = exp( -4*ln(2) * (theta / theta_3db)^2 )
        Gain_Az = exp(-4*log(2) * (angle_az / deg2rad(cfg.antenna.beamwidth_azimuth_deg))^2);
        Gain_El = exp(-4*log(2) * (angle_el / deg2rad(cfg.antenna.beamwidth_elevation_deg))^2);
        Gain_Total_Lin = 10^(cfg.antenna.gain_dbi/10) * Gain_Az * Gain_El;
        
        % Amplitude du signal reçu (Equation Radar simplifiée proportionnelle à RCS et Gain)
        % A ~ sqrt(RCS) * G / R^2 (Amplitude voltage, donc 1/R^2 car puissance en 1/R^4)
        Amp_k = sqrt(Tgt.rcs) * Gain_Total_Lin / (Dist_Tx^2);
        
        % --- B. Calcul des Distances Rx (Interférométrie) ---
        % Approximation champ lointain pour la phase différentielle
        Dist_Rx1 = Dist_Tx + dot(Vec_R/Dist_Tx, Offset_Rx1);
        Dist_Rx2 = Dist_Tx + dot(Vec_R/Dist_Tx, Offset_Rx2);
        
        % Retards aller-retour (Tau)
        tau_1 = (Dist_Tx + Dist_Rx1) / c;
        tau_2 = (Dist_Tx + Dist_Rx2) / c;
        
        % --- C. Génération du Signal de Battement (Beat Signal) ---
        
        % Terme de phase constant (Doppler/Distance) - C'est lui qui fait l'image SAR!
        % Phi = 2 * pi * fc * tau
        Phi_fixe_1 = 2 * pi * cfg.radar.fc * tau_1;
        Phi_fixe_2 = 2 * pi * cfg.radar.fc * tau_2;
        
        % Fréquence de battement f_b = K * tau
        fb_1 = K_slope * tau_1;
        fb_2 = K_slope * tau_2;
        
        % Terme RVP (Residual Video Phase) - Optionnel mais précis
        % Phi_RVP = - pi * K * tau^2
        RVP_1 = - pi * K_slope * tau_1^2;
        RVP_2 = - pi * K_slope * tau_2^2;
        
        % --- Synthèse des Signaux Analytiques ---
        
        % RAMPE MONTANTE (UP) : f_beat positive
        % S_up = exp( j * (2*pi*fb*t + Phi_fixe + RVP) )
        %Sig_Up_1 = Amp_k * exp(1j * (2*pi*fb_1*time_axis + Phi_fixe_1 + RVP_1));
        %Sig_Up_2 = Amp_k * exp(1j * (2*pi*fb_2*time_axis + Phi_fixe_2 + RVP_2));
        Sig_Up_1 = Amp_k * exp(1j * (2*pi*fb_1*time_axis + Phi_fixe_1));
        Sig_Up_2 = Amp_k * exp(1j * (2*pi*fb_2*time_axis + Phi_fixe_2));

        % RAMPE DESCENDANTE (DOWN) : K devient -K
        % Cela inverse le signe de fb et de RVP, mais PAS de Phi_fixe (terme porteur)
        % Note: Dans le hard, le signal est souvent conjugué spectralement.
        % Ici on simule physiquement : exp( j * (2*pi*(-fb)*t + Phi_fixe - RVP) )
        %Sig_Down_1 = Amp_k * exp(1j * (2*pi*(-fb_1)*time_axis + Phi_fixe_1 - RVP_1));
        %Sig_Down_2 = Amp_k * exp(1j * (2*pi*(-fb_2)*time_axis + Phi_fixe_2 - RVP_2));
        Sig_Down_1 = Amp_k * exp(1j * (2*pi*(-fb_1)*time_axis + Phi_fixe_1));
        Sig_Down_2 = Amp_k * exp(1j * (2*pi*(-fb_2)*time_axis + Phi_fixe_2));
        
        % Accumulation
        Signal_Acc_UP_Rx1   = Signal_Acc_UP_Rx1 + Sig_Up_1;
        Signal_Acc_UP_Rx2   = Signal_Acc_UP_Rx2 + Sig_Up_2;
        Signal_Acc_DOWN_Rx1 = Signal_Acc_DOWN_Rx1 + Sig_Down_1;
        Signal_Acc_DOWN_Rx2 = Signal_Acc_DOWN_Rx2 + Sig_Down_2;
    end
    
    % Ajout du Bruit Thermique (Noise Floor)
    % Calcul simplifié basé sur le SNR et la puissance moyenne
    Noise_Pwr = 10^(-cfg.radar.noise_figure_db/10) * mean(abs(Signal_Acc_UP_Rx1).^2) * 0.01; % Facteur arbitraire pour l'exemple
    %Noise = sqrt(Noise_Pwr/2) * (randn(1, N_samples) + 1j*randn(1, N_samples));
    Noise = 0;
    
    % Stockage
    RawData_UP(:, m, 1) = Signal_Acc_UP_Rx1 + Noise;
    RawData_UP(:, m, 2) = Signal_Acc_UP_Rx2 + Noise; % Bruit décorrélé normalement, ici même bruit pour simplifier (à changer si besoin)
    RawData_DOWN(:, m, 1) = Signal_Acc_DOWN_Rx1 + Noise;
    RawData_DOWN(:, m, 2) = Signal_Acc_DOWN_Rx2 + Noise;
end

toc;
fprintf('Simulation terminée.\n');

%% 4. Sauvegarde et Visualisation Rapide
save('output/KMC4_RawData.mat', 'RawData_UP', 'RawData_DOWN', 'cfg', 'Pos_Radar');

% Visualisation d'un chirp (Canal 1, Milieu de trace)
figure('Name', 'K-MC4 Simulation Check');
subplot(2,2,1);
plot(time_axis*1e3, real(RawData_UP(:, round(N_pulses/2), 1)));
title({'Signal Brut (Beat Signal) - Up Chirp', 'Note: Fréquence constante = Normal en FMCW'});
xlabel('Temps (ms)'); ylabel('Amplitude (V)');
grid on;

subplot(2,2,2);
% FFT pour vérifier la présence des cibles
L = N_samples;
Y = fft(RawData_UP(:, round(N_pulses/2), 1));
P2 = abs(Y/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);
f = (cfg.radar.adc_sample_rate/1000) * (0:(L/2))/L; % kHz (correspond aux mètres via la pente)

% Calcul fréquence théorique cible 1 pour validation visuelle
if ~isempty(cfg.scene.targets)
    R_th = norm(cfg.scene.targets(1).pos - [0 0 0]); % Approx distance
    fb_th = (cfg.radar.bandwidth / cfg.modulation.sweep_time) * (2 * R_th / 3e8);
    fprintf('Validation: Cible à %.2fm -> Fréquence de battement attendue = %.2f kHz\n', R_th, fb_th/1000);
end

plot(f, P1);
title('Spectre (Range Profile) - Up Chirp');
xlabel('Fréquence Battement (kHz)'); ylabel('|P1(f)|');
grid on;

% Spectrogramme (RTI - Range Time Intensity non focalisé)
subplot(2,2,[3 4]);
imagesc(1:N_pulses, f, db(abs(fft(RawData_UP(:,:,1)))));
ylim([0 max(f)/2]); % Zoom sur la bande utile
title('Données Brutes (Range-Compressed / RTI)');
xlabel('Numéro Impulsion (Slow Time)');
ylabel('Fréquence (Fast Time)');
colorbar;
colormap('jet');

fprintf('Données sauvegardées dans KMC4_RawData.mat\n');