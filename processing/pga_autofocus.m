function [corrected_image, total_phase_error] = pga_autofocus(sar_image, iterations, window_width)
    % Implémentation de l'algorithme Phase Gradient Autofocus (PGA).
    
    if nargin < 2, iterations = 20; end % Augmenter les itérations pour Gauss-Markov
    [num_range, num_azimuth] = size(sar_image);
    
    if nargin < 3 || isempty(window_width)
        window_width = floor(num_azimuth / 2);
    end

    corrected_image = double(sar_image);
    total_phase_error = zeros(1, num_azimuth);

    for i = 1:iterations
        % --- 1. Circular Shifting ---
        % On aligne le pic d'énergie au centre pour chaque ligne
        [~, max_indices] = max(abs(corrected_image), [], 2);
        shifted_image = zeros(size(corrected_image));
        center_idx = floor(num_azimuth / 2) + 1;
        
        for r = 1:num_range
            shift_amount = center_idx - max_indices(r);
            shifted_image(r, :) = circshift(corrected_image(r, :), shift_amount);
        end

        

        % --- 2. Fenêtrage (Windowing) ---
        current_window = max(floor(window_width / (1.5^(i-1))), 16); % Réduction plus douce
        win = zeros(1, num_azimuth);
        
        start_win = center_idx - floor(current_window / 2);
        end_win = start_win + current_window - 1;
        
        start_win = max(1, start_win);
        end_win = min(num_azimuth, end_win);
        win(start_win:end_win) = 1;
        
        windowed_image = bsxfun(@times, shifted_image, win);
        % figure(); 
        % imagesc(abs(windowed_image)); colormap hot; axis image;
        % title(sprintf('Windowed image iteration n° %d (window size = %d)', i, current_window));
        % xlabel('Azimut'); ylabel('Range');


        % --- 3. Phase Gradient Estimation (Domaine Doppler) ---
        % CORRECTION: Utilisation de FFT pour aller vers le domaine Doppler
        % ifftshift déplace le centre (N/2) vers l'indice 1 pour éviter la rampe linéaire
        g_n = fft(ifftshift(windowed_image, 2), [], 2);
        % Approximation de la dérivée
        g_n_diff = [diff(g_n, 1, 2), g_n(:, 1) - g_n(:, end)];
        
        % Estimateur Maximum de Vraisemblance (Linear Unbiased Minimum Variance)
        num = sum(imag(conj(g_n) .* g_n_diff), 1);
        den = sum(abs(g_n).^2, 1) + 1e-10; % Petit epsilon pour éviter div/0
        phase_gradient = num ./ den;

        % --- 4. Intégration et Detrend ---
        phase_est = cumsum(phase_gradient);
        
        % Suppression de la tendance linéaire (positionnement) et constante
        x = 1:num_azimuth;
        p = polyfit(x, phase_est, 1);
        phase_est = phase_est - polyval(p, x);
        
        total_phase_error = total_phase_error + phase_est;

        % --- 5. Correction de l'image ---
        % CORRECTION: FFT pour aller en Doppler, IFFT pour revenir
        image_freq = fft(corrected_image, [], 2);
        correction_term = exp(-1j * phase_est);
        
        % Application de la correction
        corrected_image = ifft(bsxfun(@times, image_freq, correction_term), [], 2);
    end
end