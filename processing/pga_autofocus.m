function [img, total_phase_error] = pga_autofocus(sar_image, iterations, window_width)
% PGA_AUTOFOCUS Implements the Phase Gradient Autofocus (PGA) algorithm.
%
%   [corrected_image, phase_error] = pga_autofocus(sar_image, iterations, window_width)
%
%   Args:
%       sar_image (complex double): Focused SAR image (but blurred).
%                                   Format: [Range, Azimuth] (Rows, Cols).
%       iterations (int): Number of iterations. Default is 5.
%       window_width (int): Initial window width. If empty, takes N_az/2.
%
%   Returns:
%       corrected_image (complex double): Sharp image.
%       phase_error (double): final estimated phase error.

    if nargin < 3 || isempty(window_width)
        [~, N_az] = size(sar_image);
        window_width = floor(N_az / 2);
    end
    if nargin < 2 || isempty(iterations)
        iterations = 5;
    end

    img = sar_image;

    [N_range, N_az] = size(img);



    % Phase d'erreur cumulée (pour le diagnostic)
    total_phase_error = zeros(1, N_az);

    % Calculer l'entropie initiale
    initial_entropy = calculate_entropy(img);
    fprintf('Entropie initiale: %.4f\n', initial_entropy);

    for i = 1:iterations
        % 1. CENTER SHIFTING (Alignement des points brillants)
        % On trouve le max sur chaque ligne (Range)
        [~, max_indices] = max(abs(img), [], 2);

        % On décale circulairement chaque ligne pour mettre le max au centre
        shifted_img = zeros(size(img));
        center_idx = ceil(N_az / 2);

        for r = 1:N_range
            % shift = (N_az // 2) - max_indices[r]
            shift_val = center_idx - max_indices(r);
            shifted_img(r, :) = circshift(img(r, :), shift_val, 2);
        end

        % --- Visualisation du Shift (Debug) ---
        if i == 1
            figure(99);
            imagesc(abs(shifted_img));
            title('PGA: Image alignée (Circular Shift - Iter 1)');
            xlabel('Azimuth'); ylabel('Range');
            colorbar;
            drawnow;
        end

        % 2. WINDOWING (Fenêtrage)
        % On réduit la fenêtre à chaque itération pour exclure le bruit
        current_width = floor(window_width / i);

         if i < 6
             current_width = 200; 
         end   
        display(current_width)

        % Replication of logic from python source (commented out hard overrides)
        % if i < 8 % i is 1-based here, python was < 7 (0-based)
        %     current_width = 60;
        % end
        % current_width = 2; % Found in python source, likely debug
        
        % Ensure width is at least 2 and even for simplicity
        if current_width < 2
            current_width = 2;
        end
        
        start_idx = max(1, center_idx - floor(current_width / 2));
        end_idx = min(N_az, center_idx + floor(current_width / 2));
        
        window = zeros(1, N_az);
        window(start_idx:end_idx) = 1.0;

        % Application de la fenêtre sur toutes les lignes
        % Broadcasting window (1, N_az) across rows (N_range, N_az)
        windowed_img = shifted_img .* window;

        % 3. PASSAGE DOMAINE DONNÉES (FFT)
        % On passe dans le domaine fréquentiel (Doppler/Temps lent)
        G = fft(windowed_img, [], 2);

        % 4. ESTIMATION DU GRADIENT (Produit conjugué + Somme)
        % Différence de phase entre k et k-1
        % Python: G[:, 1:] * np.conj(G[:, :-1])
        % Matlab: G(:, 2:end) .* conj(G(:, 1:end-1))
        term = G(:, 2:end) .* conj(G(:, 1:end-1));
        
        % On somme sur l'axe Range (axis=0 in Py -> dimension 1 in Mat) pour moyenner le bruit
        numerator = sum(term, 1);

        % On extrait la phase (le gradient)
        dphi = angle(numerator);

        % On remet le premier échantillon à 0 (pas de gradient au début)
        dphi = [0, dphi];

        % 5. INTÉGRATION (Retrouver l'erreur de phase)
        estimated_error = cumsum(dphi);

        % On enlève la tendance linéaire
        x_axis = 1:N_az;
        p = polyfit(x_axis, estimated_error, 1);
        estimated_error = estimated_error - polyval(p, x_axis);

        % Borner l'erreur entre -pi et pi (Wrap phase)
        estimated_error = angle(exp(1j * estimated_error));

        % Mise à jour de l'erreur totale
        total_phase_error = total_phase_error + estimated_error;
        % Re-wrap l'erreur totale aussi
        total_phase_error = angle(exp(1j * total_phase_error));
        
        % 6. CORRECTION DE L'IMAGE
        % On applique la correction dans le domaine fréquentiel de l'image ORIGINALE (non shiftée)
        IMG_original_freq = fft(img, [], 2);

        % Correction : on multiplie par e^(-j * erreur)
        correction_phasor = exp(1j * estimated_error);
        
        % Broadcasting correction
        IMG_corrected = IMG_original_freq .* correction_phasor;

        % Retour domaine image
        img = ifft(IMG_corrected, [], 2);

        % Calculer et afficher l'entropie après correction
        current_entropy = calculate_entropy(img);
        fprintf('Itération %d/%d: entropie: %.4f\n', i, iterations, current_entropy);
    end
end

function entropy = calculate_entropy(image)
    % Calcule l'entropie de l'image basée sur l'histogramme des amplitudes.
    % Une entropie plus faible indique généralement une image plus nette.
    
    % Prendre l'amplitude de l'image complexe
    amplitude = abs(image);

    % Normaliser
    max_amp = max(amplitude(:));
    if max_amp == 0
        entropy = 0;
        return;
    end
    amplitude_norm = amplitude / max_amp;

    % Quantification sur 256 niveaux
    % Create bins: 0 to 255
    quantized = floor(amplitude_norm * 255);
    quantized(quantized > 255) = 255; 

    % Calculer l'histogramme
    % Matlab histcounts returns counts in bins. 
    % We want bins for integers 0, 1, ..., 255.
    % Bin edges: -0.5, 0.5, ..., 255.5 works well for integers.
    edges = -0.5:1:255.5;
    hist_counts = histcounts(quantized(:), edges);
    
    % Probability density
    p = hist_counts / sum(hist_counts);

    % Supprimer les bins vides pour éviter log(0)
    p = p(p > 0);

    % Calculer l'entropie: H = -sum(p * log2(p))
    entropy = -sum(p .* log2(p));
end
