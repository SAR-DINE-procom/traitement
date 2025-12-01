% filepath: c:\code\py\sar\getParams.m
function [fc, rangeResolution, crossRangeResolution, bw, prf, aperture, tpd, fs, speed, flightDuration, maxRange, targetsPosition, motionErrors] = getParams(jsonPath)
    if ~isfile(jsonPath)
        error('Le fichier de configuration est introuvable : %s', jsonPath);
    end

    fid = fopen(jsonPath); 
    raw = fread(fid, inf); 
    str = char(raw'); 
    fclose(fid); 

    config = jsondecode(str);

    fc = config.fc;
    rangeResolution = config.rangeResolution;
    crossRangeResolution = config.crossRangeResolution;
    bw = config.bw;
    prf = config.prf;
    aperture = config.aperture;
    tpd = config.tpd;
    fs = config.fs;
    speed = config.speed;
    flightDuration = config.flightDuration;
    maxRange = config.maxRange;
        if isfield(config, 'motionErrors')
        motionErrors = config.motionErrors;
    else
        motionErrors = [0; 0; 0]; % Pas d'erreur par défaut
    end

    if isfield(config, 'targetsPosition')
        targetsPosition = config.targetsPosition'; 
    else
        targetsPosition = [];
    end
end