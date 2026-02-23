function [corrected_image, total_phase_error, rms_history] = pga_autofocus(sar_image, iterations)
    [num_range, num_az] = size(sar_image);
    corrected_image = sar_image;
    total_phase_error = zeros(1, num_az);
    rms_history = zeros(1, iterations);
    
    % Sélection des bins les plus forts
    num_bins_to_use = min(500, num_range);
    W = num_az; 
    center_idx = floor(num_az / 2) + 1;

    for iter = 1:iterations
        % Sélection basée sur l'énergie
        energy = sum(abs(corrected_image).^2, 2);
        [~, sorted_idx] = sort(energy, 'descend');
        sel_idx = sorted_idx(1:num_bins_to_use);
        sub_image = corrected_image(sel_idx, :);

        % 1. Circular Shifting 
        [~, max_pos] = max(abs(sub_image), [], 2);
        shifted_img = zeros(num_bins_to_use, num_az);
        for r = 1:num_bins_to_use
            shifted_img(r, :) = circshift(sub_image(r, :), center_idx - max_pos(r));
        end

        % 2. Windowing progressif (20% par itération, min 5px)
        if iter > 1, W = floor(W * 0.8); end
        W = max(W, 5); 
        
        win = zeros(1, num_az);
        w_start = max(1, center_idx - floor(W/2));
        w_end = min(num_az, center_idx + floor(W/2));
        win(w_start:w_end) = 1;
        g_n = bsxfun(@times, shifted_img, win);

        % 3. Estimation LUMV
        % L'ASTUCE MATHEMATIQUE EST ICI :
        % ifftshift(g_n, 2) annule la rampe de phase linéaire avant la FFT.
        % On fait ensuite un fftshift pour réaligner l'axe de fréquence avec l'image globale.
        G_n = fftshift(fft(ifftshift(g_n, 2), [], 2), 2);
        G_n_dot = [diff(G_n, 1, 2), zeros(num_bins_to_use, 1)];
        
        num = sum(imag(conj(G_n) .* G_n_dot), 1);
        den = sum(abs(G_n).^2, 1) + 1e-12;
        phi_dot = num ./ den;

        % 4. Intégration et Correction
        phi_est = cumsum(phi_dot);
        
        % Suppression de la tendance linéaire pour empêcher l'image de dériver
        x_ax = 1:num_az;
        p = polyfit(x_ax, phi_est, 1);
        phi_est = phi_est - polyval(p, x_ax);
        
        % Application de la correction
        img_freq = fftshift(fft(corrected_image, [], 2), 2);
        correction = exp(-1j * phi_est);
        corrected_image = ifft(ifftshift(bsxfun(@times, img_freq, correction), 2), [], 2);
        
        total_phase_error = total_phase_error + phi_est;
        rms_history(iter) = sqrt(mean(phi_est.^2));
    end
end