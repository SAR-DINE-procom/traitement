function [corrected_image, total_phase_error] = pga_autofocus(sar_image, iterations, window_width)
    % Implémentation de l'algorithme Phase Gradient Autofocus (PGA).
    
    if nargin < 2, iterations = 5; end
    [num_range, num_azimuth] = size(sar_image);
    
    if nargin < 3 || isempty(window_width)
        window_width = floor(num_azimuth / 2);
    end

    corrected_image = double(sar_image);
    total_phase_error = zeros(1, num_azimuth);

    for i = 1:iterations
        % --- 1. Circular Shifting (Centrage sur le point le plus brillant) ---
        [~, max_indices] = max(abs(corrected_image), [], 2);
        shifted_image = zeros(size(corrected_image));
        center_idx = floor(num_azimuth / 2) + 1;
        
        for r = 1:num_range
            shift_amount = center_idx - max_indices(r);
            shifted_image(r, :) = circshift(corrected_image(r, :), shift_amount);
        end

        % --- 2. Fenêtrage (Windowing) ---
        current_window = max(floor(window_width / (2^(i-1))), 10);
        win = zeros(1, num_azimuth);
        
        start_win = center_idx - floor(current_window / 2);
        end_win = start_win + current_window - 1;
        
        % Protection des indices
        start_win = max(1, start_win);
        end_win = min(num_azimuth, end_win);
        win(start_win:end_win) = 1;
        
        % Application de la fenêtre sur chaque ligne (Broadcasting)
        windowed_image = bsxfun(@times, shifted_image, win);

        % --- 3. Phase Gradient Estimation (Domaine Doppler) ---
        % ifftshift recentre les fréquences pour l'IFFT
        g_n = ifft(ifftshift(windowed_image, 2), [], 2);
        
        % Approximation de la dérivée (différence finie)
        g_n_diff = [diff(g_n, 1, 2), g_n(:, 1) - g_n(:, end)];
        
        % Estimateur Maximum de Vraisemblance du gradient
        num = sum(imag(conj(g_n) .* g_n_diff), 1);
        den = sum(abs(g_n).^2, 1);
        phase_gradient = num ./ den;

        % --- 4. Intégration et Detrend (Suppression de la rampe linéaire) ---
        phase_error = cumsum(phase_gradient);
        
        x = 1:num_azimuth;
        p = polyfit(x, phase_error, 1);
        phase_error = phase_error - polyval(p, x);
        
        total_phase_error = total_phase_error + phase_error;

        % --- 5. Correction de l'image ---
        image_freq = ifft(corrected_image, [], 2);
        correction_term = exp(-1j * phase_error);
        
        % Application de la phase et retour en spatial
        corrected_image = fft(bsxfun(@times, image_freq, correction_term), [], 2);
    end
end