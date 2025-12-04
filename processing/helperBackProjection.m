function data = helperBackProjection(sigdata,rnggrid,fastTime,fc,fs,prf,speed,crossRangeResolution,c)
    
    [~, numPulses] = size(sigdata);
    
    % --- DEFINITION DE LA GRILLE SPATIALE (2.5 cm) ---
    gridStep = 0.025; % 2.5 cm
    
    % Définition des limites de la zone à imager (en mètres)
    % Vous pouvez ajuster ces bornes selon vos besoins
    rangeLims = [0 10];       % De 0 à 10m en distance
    crossRangeLims = [0 10];  % De 0 à 10m en azimut (ajustez selon la longueur du vol)
    
    % Création des axes de la grille image
    imageRangeAxis = rangeLims(1):gridStep:rangeLims(2);
    imageCrossRangeAxis = crossRangeLims(1):gridStep:crossRangeLims(2);
    
    % Initialisation de l'image finale
    data = zeros(length(imageRangeAxis), length(imageCrossRangeAxis));
    
    % Axe azimutal de la trajectoire radar (pour le calcul géométrique)
    radarAzimuthPos = (0:numPulses-1) * (speed/prf); 

    % --- BOUCLE DE RETROPROJECTION ---
    % On itère sur chaque pixel de l'image finale
    for i = 1:length(imageRangeAxis)
        
        pixelRange = imageRangeAxis(i);
        
        % Calcul de l'ouverture synthétique nécessaire pour ce pixel (Théorie)
        lsynth = (c/fc) * pixelRange / (2*crossRangeResolution);
        lsar = round(lsynth / (speed/prf));
        lsar = lsar + mod(lsar,2); % Impair
        hn = hanning(lsar).';
        
        for j = 1:length(imageCrossRangeAxis)
            
            pixelCrossRange = imageCrossRangeAxis(j);
            
            % Trouver l'index du radar le plus proche de la position azimutale du pixel
            % (C'est le centre de l'ouverture synthétique)
            [~, centerPulseIdx] = min(abs(radarAzimuthPos - pixelCrossRange));
            
            k_start = centerPulseIdx - floor(lsar/2);
            k_end   = centerPulseIdx + floor(lsar/2);
            
            count = 0;
            
            % Sommation cohérente sur l'ouverture
            for k = k_start : k_end
                if k >= 1 && k <= numPulses
                    radarPosAtK = radarAzimuthPos(k);
                    
                    % Distance aller-retour exacte entre le radar(k) et le pixel(i,j)
                    distTwoWay = sqrt((radarPosAtK - pixelCrossRange)^2 + pixelRange^2) * 2;
                    td = distTwoWay / c;
                    
                    % Conversion en index d'échantillon (Fast Time)
                    cell = round(td * fs) + 1;
                    
                    if cell >= 1 && cell <= size(sigdata, 1)
                        signal = sigdata(cell, k);
                        h_idx = k - k_start + 1;
                        
                        % Matched Filter (Correction de phase)
                        if h_idx >= 1 && h_idx <= length(hn)
                            count = count + hn(h_idx) * signal * exp(1j * 2 * pi * fc * td);
                        end
                    end
                end
            end
            data(i,j) = count;
        end
    end
end
