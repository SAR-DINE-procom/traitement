%% Simulation SAR 2D : Impact du Bras de Levier et Autofocus
clear; clc; close all;

% --- 1. Paramètres ---
N = 256;                % Taille de l'image (Pixels)
fc = 24e9; lambda = 3e8 / fc;
L = [0.04; 0.1; 0.1];   % Bras de levier
u_local = [0.89; 0; 0.45];    % Visée LOS
fs = 100; t = (0:N-1)'/fs;
errorType = 'Gauss-Markov'; 
%errorType = 'Sinus'; 
angles = zeros(N, 3);

if strcmp(errorType, 'Gauss-Markov')
    tau = 0.8; sigma = deg2rad(0.6);
    dt = 1/fs; coeff = exp(-dt/tau);
    for k = 2:N
        angles(k,:) = coeff * angles(k-1,:) + sigma*sqrt(1-coeff^2)*randn(1,3);
    end
elseif strcmp(errorType, 'Sinus')    
    A = 5; %Marche pas 
    f = 0.5; 
    for k = 2:N
        angles(k,1) = A * sin(2*pi*f*k/fs);
    end
end 


% --- 3. Calcul de l'erreur de phase 1D (Azimut) ---
DR = zeros(N, 1);
for k = 1:N
    phi = angles(k,1); theta = angles(k,2); psi = angles(k,3);
    Rrot = [cos(psi)*cos(theta), cos(psi)*sin(theta)*sin(phi)-sin(psi)*cos(phi), cos(psi)*sin(theta)*cos(phi)+sin(psi)*sin(phi);
            sin(psi)*cos(theta), sin(psi)*sin(theta)*sin(phi)+cos(psi)*cos(phi), sin(psi)*sin(theta)*cos(phi)-cos(psi)*sin(phi);
            -sin(theta),         cos(theta)*sin(phi),                         cos(theta)*cos(phi)];
    DR(k) = dot(u_local, Rrot*L - L);
end
phase_error = (4*pi/lambda) * DR; % Vecteur N x 1

% --- 4. Création de l'Image Originale (Points Brillants) ---
img_ideal = zeros(N, N);
img_ideal(128, 128) = 1; 
img_ideal(60, 80) = 0.7; 
img_ideal(180, 200) = 0.9;

% --- 5. Application de l'erreur dans le domaine History ---
% L'erreur d'attitude affecte principalement l'axe Azimut (colonnes)
history_ideal = fftshift(fft2(img_ideal));
% On applique la même erreur de phase sur chaque ligne de range (pour chaque instant t)
error_matrix = exp(1i * phase_error); % N x 1
history_degraded = history_ideal .* repmat(error_matrix, 1, N);

img_degraded = ifft2(ifftshift(history_degraded));

% --- 6. Autofocus (Simulation de l'appel) ---
[img_corrected, total_phase_error] = pga_autofocus(img_degraded.', 5); 

% --- 7. Visualisation 2D ---
figure('Color', 'w', 'Position', [50, 50, 1200, 400]);

subplot(1,3,1);
max = max(20 * log10(abs(img_ideal.'))); 
imagesc(20 * log10(abs(img_ideal.')) - max); colormap hot; axis image;
title('\bf 1. Image Originale (Idéale)');
xlabel('Azimut'); ylabel('range');
colorbar;

subplot(1,3,2);
imagesc(abs(img_degraded.')); colormap hot; axis image;
title('\bf 2. Image Dégradée (Erreurs Attitude)');
xlabel('Azimut'); ylabel('Range');
colorbar;

subplot(1,3,3);
imagesc(abs(img_corrected.')); colormap hot; axis image;
colorbar;
hold on; 
plot(128, 128, 'rx'); 
plot(80, 60, 'rx'); 
plot(200, 180, 'rx'); 
hold off; 
title('\bf 3. Image Corrigée (PGA)');
xlabel('Azimut'); ylabel('Range');

% Ajout d'une coupe 1D pour bien voir la focalisation
figure('Color', 'w');
plot(abs(img_ideal(:,128)), 'k--'); hold on;
plot(abs(img_degraded(:,128)), 'r--');
plot(abs(img_corrected(:,128)), 'g--', 'LineWidth', 1.5);
title('Coupe Azimut sur le point central');
legend('Idéal', 'Dégradé', 'Corrigé'); grid on;


figure('Color', 'w'); 
plot(abs(phase_error), 'r--'); hold on;
plot(abs(total_phase_error), 'g'); 
legend('Réelle', 'Estimée')
