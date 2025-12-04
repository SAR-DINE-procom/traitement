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
    waveform = phased.LinearFMWaveform('SampleRate', fs, 'PulseWidth', tpd, 'PRF', prf, 'SweepBandwidth', bw);
    radarPlatform = phased.Platform('InitialPosition', [0;0;2], 'Velocity', [0; speed; 0]);
    
    antenna = phased.CosineAntennaElement('FrequencyRange', [20e9 26e9]);
    antennaGain = aperture2gain(aperture, c/fc); 
    
    transmitter = phased.Transmitter('PeakPower', 50e3, 'Gain', antennaGain);
    radiator = phased.Radiator('Sensor', antenna, 'OperatingFrequency', fc, 'PropagationSpeed', c);
    collector = phased.Collector('Sensor', antenna, 'PropagationSpeed', c, 'OperatingFrequency', fc);
    receiver = phased.ReceiverPreamp('SampleRate', fs, 'NoiseFigure', 30);
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
    truncrangesamples = ceil((2*maxRange/c)*fs);
    
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
        targetAngle(1,:) = refangle;
        
        sig = radiator(sig, targetAngle);
        sig = channel(sig, radarpos_real, targetpos, radarvel, targetvel);
        sig = target(sig);
        sig = collector(sig, targetAngle);
        rxsig(:,ii) = receiver(sig);
    end

    % --- Visualization: Raw Data ---
    figure(3);
    imagesc(real(rxsig));
    title('Raw Radar Data (Real Part)');
    xlabel('Pulse Index (Slow Time)');
    ylabel('Fast Time Sample');
    colorbar;

    % --- Signal Processing ---
    % 1. Pulse Compression
    pulseCompression = phased.RangeResponse('RangeMethod', 'Matched filter', 'PropagationSpeed', c, 'SampleRate', fs);
    matchingCoeff = getMatchedFilter(waveform);
    [cdata, rnggrid] = pulseCompression(rxsig, matchingCoeff);

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

    % 2. Backprojection Algorithm
    fastTime = (0:1/fs:(truncrangesamples-1)/fs);
    
    % Appel de la fonction modifiée
    bpa_processed = helperBackProjection(cdata, rnggrid, fastTime, fc, fs, prf, speed, crossRangeResolution, c);

    % --- Visualization ---
    figure(2);
    
    % Il faut recréer les axes utilisés dans helperBackProjection pour l'affichage
    gridStep = 0.025;
    rangeLims = [0 10];
    crossRangeLims = [0 10];
    
    imgRangeAxis = rangeLims(1):gridStep:rangeLims(2);
    imgCrossAxis = crossRangeLims(1):gridStep:crossRangeLims(2);
    
    [~, numPulses] = size(cdata);
    azimuthDist = (0:numPulses-1) * (speed/prf); 
    
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

    % --- Export Data ---
    if ~exist('output', 'dir'), mkdir('output'); end
    save('output/out.mat', 'cdata', 'real_path', 'ideal_path');
end
