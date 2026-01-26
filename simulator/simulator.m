% filepath: c:\code\py\sar\simulator\simulator.m
function simulator(configPath)
    % --- Path and Argument Management ---
    currentScriptPath = fileparts(mfilename('fullpath'));
    addpath(fullfile(currentScriptPath, '..')); % Add parent dir for helpers
    
    if nargin < 1
        configPath = fullfile(currentScriptPath, '..', 'conf', 'config.json');
    end

    % --- Load Configuration ---
    c = physconst('LightSpeed');
    [fc, rangeResolution, crossRangeResolution, bw, prf, aperture, tpd, fs, speed, flightDuration, maxRange, targetsPosition, motionErrors] = getParams(configPath); 

    % --- Radar System Setup ---
    waveform = phased.LinearFMWaveform('SampleRate', fs, 'PulseWidth', tpd, 'PRF', prf, 'SweepBandwidth', bw, SweepInterval = 'Symmetric');
    %waveform = phased.FMCWWaveform('SampleRate', fs, 'SweepTime', 1/prf, 'SweepBandwidth', bw, 'SweepDirection', 'Triangle');
    plot(waveform)
    radarPlatform = phased.Platform('InitialPosition', [0;0;2], 'Velocity', [0; speed; 0]);
    
    antenna = phased.CosineAntennaElement('FrequencyRange', [20e9 26e9]);
    antennaGain = aperture2gain(aperture, c/fc); 
    
    transmitter = phased.Transmitter('PeakPower', 50e3, 'Gain', antennaGain);
    radiator = phased.Radiator('Sensor', antenna, 'OperatingFrequency', fc, 'PropagationSpeed', c);
    collector = phased.Collector('Sensor', antenna, 'PropagationSpeed', c, 'OperatingFrequency', fc);
    receiver = phased.ReceiverPreamp('SampleRate', fs, 'NoiseFigure', 3); % , 'Gain', 20 + antennaGain
    channel = phased.FreeSpace('PropagationSpeed', c, 'OperatingFrequency', fc, 'SampleRate', fs, 'TwoWayPropagation', true);

    % --- Target Setup ---
    if isempty(targetsPosition)
        warning('No targets in JSON. Using default.');
        targetpos = [2; 0; 0];
    else
        targetpos = targetsPosition;
    end
    
    [~, numTargets] = size(targetpos);
    targetvel = zeros(3, numTargets); 
    target = phased.RadarTarget('OperatingFrequency', fc, 'MeanRCS', ones(1, numTargets));
    pointTargets = phased.Platform('InitialPosition', targetpos, 'Velocity', targetvel);

    % --- Ground Truth Visualization ---
    figure(1); h = axes;
    plot(targetpos(2,:), targetpos(1,:), '*g', 'MarkerSize', 10); 
    set(h, 'Ydir', 'reverse');
    xlim([-10 10]); 
    minR = min(targetpos(1,:));
    maxR = max(targetpos(1,:));
    ylim([max(0, minR-5) maxR+5]); 
    title('Ground Truth'); ylabel('Range'); xlabel('Cross-Range');

    % --- Simulation Loop ---
    slowTime = 1/prf;
    numpulses = flightDuration/slowTime + 1;
    maxTime = (2*maxRange/c) + tpd; 
    truncrangesamples = ceil(maxTime * fs);
    
    refangle = zeros(1, size(targetpos, 2));
    rxsig = zeros(truncrangesamples, numpulses);
    
    ideal_path = zeros(3, numpulses);
    real_path = zeros(3, numpulses);

    for ii = 1:numpulses
        % 1. Update Kinematics
        [radarpos_ideal, radarvel] = radarPlatform(slowTime);
        [targetpos, targetvel] = pointTargets(slowTime);
        
        % 2. Apply Motion Errors
        perturbation = randn(3,1) .* motionErrors; 
        %perturbation = 2 * sin(2 * pi * ii) ^ 2; 
        
        perturbation_x = 0.001 * sin(2*pi*2*ii);
        radarpos_real = radarpos_ideal + perturbation;
        
        % 3. Store Paths
        ideal_path(:, ii) = radarpos_ideal;
        real_path(:, ii) = radarpos_real;
        
        % 4. Physics Simulation (using Real Position)
        [targetRange, targetAngle] = rangeangle(targetpos, radarpos_real);
        
        sig = waveform();
        sig = sig(1:truncrangesamples);
        sig = transmitter(sig);
        
        % Force spotlight/tracking mode (comment out for stripmap)
        %targetAngle(1,:) = refangle;
        %fprintf('Pulse %d: targetAngle(1,1) = %.4f rad\n', ii, targetAngle(1,1));
        sig = radiator(sig, targetAngle);
        sig = channel(sig, radarpos_real, targetpos, radarvel, targetvel);
        sig = target(sig);
        sig = collector(sig, targetAngle);
        rxsig(:,ii) = receiver(sig);

    end
    figure(2);
    plot(abs(rxsig(:, 1)));
    title('First Pulse Response');
    grid on;
    hold on; 
    % show() removed as it is not a valid MATLAB command 
    % --- Visualization: Raw Data ---
    figure(3);
    imagesc(real(rxsig));
    title('Raw Radar Data (Real Part)');
    xlabel('Pulse Index (Slow Time)');
    ylabel('Fast Time Sample');
    colorbar;

    % --- Signal Processing ---
    % 1. Pulse Compression
    %pulseCompression = phased.RangeResponse('RangeMethod', 'Matched filter', 'PropagationSpeed', c, 'SampleRate', fs);
    %matchingCoeff = getMatchedFilter(waveform);
    %[cdata, rnggrid] = pulseCompression(rxsig, matchingCoeff);
    ref_sig = waveform();
    ref_sig = ref_sig(1:size(rxsig, 1));
    % 1. Dechirp (Mélange avec la référence)
    dechirpedSig = dechirp(rxsig, ref_sig); 

    % 2. Range Processing (FFT)
    % La distance est proportionnelle à la fréquence de battement
    Nfft = 2^nextpow2(size(dechirpedSig, 1));
    cdata_fft = fft(dechirpedSig, Nfft,1);
    
    % 3. Calcul de la grille de distance (Range Grid)
    % Formule FMCW : f_beat = (2 * Slope * R) / c  => R = (c * f_beat) / (2 * Slope)
    slope = bw * prf; % Pente du chirp pour calcul de distance plus tard
    freq_bins = (0:Nfft-1) * (fs/Nfft);
    rnggrid = (c * freq_bins) / (2 * slope);
    
    % On ne garde que la partie positive et pertinente du spectre
    half_spectrum = floor(Nfft/2);
    cdata = cdata_fft(1:half_spectrum, :);
    rnggrid = rnggrid(1:half_spectrum).'; %

    % --- Visualization: Range Compressed Data ---
    figure(4);
    % Calcul de l'axe azimutal pour l'affichage
    [~, numPulses] = size(cdata);
    azimuthDist = (0:numPulses-1) * (speed/prf); 
    imagesc(azimuthDist, rnggrid, abs(cdata));
    title('Range Compressed Data (Envelope)');
    xlabel('Cross-Range (m)');
    ylabel('Range (m)');
    axis xy;
    ylim([0 10]); % Zoom sur la zone proche (cibles)
    colorbar;

    figure(5);
    plot(rnggrid, abs(cdata(:,1)));
    title('Range Compressed Signal (1st Pulse)');
    xlabel('Range (m)');
    ylabel('Amplitude');
    grid on;
    xlim([0 20]);

    % 2. Backprojection Algorithm
    fastTime = (0:1/fs:(truncrangesamples-1)/fs);
    
    % Appel de la fonction modifiée
    bpa_processed = helperBackProjection(cdata, rnggrid, fastTime, fc, fs, prf, speed, crossRangeResolution, c);

    % --- Autofocus (PGA) ---
    addpath(fullfile(currentScriptPath, '..', 'processing'));
    fprintf('Running PGA Autofocus...\n');
    [bpa_autofocused, phase_error] = pga_autofocus(bpa_processed, 20);

    % --- Visualization ---
    
    % Il faut recréer les axes utilisés dans helperBackProjection pour l'affichage
    gridStep = 0.025;
    rangeLims = [0 10];
    crossRangeLims = [0 10];
    
    imgRangeAxis = rangeLims(1):gridStep:rangeLims(2);
    imgCrossAxis = crossRangeLims(1):gridStep:crossRangeLims(2);
    
    [~, numPulses] = size(cdata);
    azimuthDist = (0:numPulses-1) * (speed/prf); 
    
    figure(6); % New figure for the final image
    subplot(1,2,1);
    imagesc(imgCrossAxis, imgRangeAxis, abs(bpa_processed));
    title('SAR Image (Backprojection)');
    xlabel('Cross-range (m)');
    ylabel('Range (m)');
    set(gca, 'YDir', 'normal'); 
    axis xy; 
    axis equal; 
    ylim([0 10]); % Zoom on near field
    xlim([0 max(azimuthDist)]);
    colorbar;

    subplot(1,2,2);
    imagesc(imgCrossAxis, imgRangeAxis, abs(bpa_autofocused));
    title('SAR Image (PGA Autofocused)');
    xlabel('Cross-range (m)');
    ylabel('Range (m)');
    set(gca, 'YDir', 'normal'); 
    axis xy; 
    axis equal; 
    ylim([0 10]); % Zoom on near field
    xlim([0 max(azimuthDist)]);
    colorbar;

    figure(7);
    plot(phase_error);
    title('Estimated Phase Error');
    xlabel('Azimuth Index');
    ylabel('Phase (rad)');

    % --- Export Data ---
    if ~exist('output', 'dir'), mkdir('output'); end
    
    % 1. Export standard .mat (pour Python/MATLAB)
    save('output/out.mat', 'cdata', 'real_path', 'ideal_path', 'rxsig');

    % 2. Export TDMS (Pour compatibilité NI DAQmx)
    tdmsFileName = fullfile(pwd, 'output', 'simulation_raw.tdms');
    
    try
        if exist(tdmsFileName, 'file')
            delete(tdmsFileName);
        end

        % Verification mémoire (Heuristique simple)
        s = whos('rxsig');
        % Limite arbitraire (ex: 200MB pour rxsig -> ~800MB requis pour la conversion)
        if s.bytes > 200*1024*1024 
            warning('Signal trop volumineux (>200MB) pour export TDMS direct. Ignoré pour éviter OOM.');
        else
            raw_vector = rxsig(:); 
            timeVec = seconds((0:length(raw_vector)-1)' / fs);
            
            T = timetable(timeVec, real(raw_vector), imag(raw_vector), ...
                'VariableNames', {'Dev1_ai0_I', 'Dev1_ai1_Q'});
            
            tdmswrite(tdmsFileName, T);
            fprintf('Export TDMS réussi : %s\n', tdmsFileName);
        end
        
    catch ME
        warning('Impossible d''écrire le fichier TDMS.');
        fprintf('Erreur : %s\n', ME.message);
        fprintf('Les données brute sont disponibles dans output/out.mat\n');
    end
end
