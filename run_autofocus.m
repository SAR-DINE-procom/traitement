%% Initialisation
clear; clc; close all;

N_range = 100; N_az = 512;
img_ideal = zeros(N_range, N_az);
img_ideal(45, 256) = 100; % Cible centrale
img_ideal(55, 280) = 70;  % Cible secondaire

%% Simulation de l'erreur (Motion Errors)
t = linspace(-1, 1, N_az);
% Erreur de phase complexe : quadratique + haute fréquence
true_phase = 10*t.^2 + 4*t.^3 + 2*sin(2*pi*4*t); 

% On applique l'erreur dans le domaine Doppler
IMG_freq = ifft(img_ideal, [], 2); 
IMG_blurred_freq = bsxfun(@times, IMG_freq, exp(1j * true_phase));
img_blurred = fft(IMG_blurred_freq, [], 2); % Image floue

%% Application PGA
% On utilise une fenêtre initiale de N_az/2 pour bien capturer le flou
[img_fixed, estimated_phase] = pga_autofocus(img_blurred, 5, N_az/2);

%% Visualisation
figure('Color', 'w', 'Name', 'Analyse PGA Autofocus');

subplot(2,2,1); imagesc(abs(img_blurred)); title('Image Floue (Entrée)');
subplot(2,2,2); imagesc(abs(img_fixed)); title('Image Corrigée (PGA)');

subplot(2,2,3);
plot(true_phase - mean(true_phase), 'r--', 'LineWidth', 2); hold on;
plot(estimated_phase, 'b');
legend('Réelle', 'Estimée'); title('Phase Error (rad)'); grid on;

subplot(2,2,4);
plot(20*log10(abs(img_blurred(45,:))/max(abs(img_blurred(:)))), 'r'); hold on;
plot(20*log10(abs(img_fixed(45,:))/max(abs(img_fixed(:)))), 'b', 'LineWidth', 1.5);
title('Profil Azimut (dB)'); ylim([-60 0]); grid on;