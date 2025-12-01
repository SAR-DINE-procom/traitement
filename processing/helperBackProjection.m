function data = helperBackProjection(sigdata,rnggrid,fastTime,fc,fs,prf,speed,crossRangeResolution,c)
    
    [~, numPulses] = size(sigdata);
    
    % Reconstruction de l'axe azimutal basé sur la vitesse et le PRF
    azimuthDist = (0:numPulses-1) * (speed/prf); 
    
    % Initialisation
    data = zeros(length(rnggrid), length(azimuthDist));
    
    % Définition des zones à traiter (on peut ajuster selon besoin)
    rangelims = [0 10]; 
    crossrangelims = [-10 10];

    rangeIdx =  [find(rnggrid>rangelims(1), 1) find(rnggrid<rangelims(2),1,'last')];
    if isempty(rangeIdx), rangeIdx = [1 length(rnggrid)]; end

    crossrangeIdxStart = find(azimuthDist>crossrangelims(1),1);
    if isempty(crossrangeIdxStart), crossrangeIdxStart = 1; end
    
    crossrangeIdxStop = find(azimuthDist<crossrangelims(2),1,'last');
    if isempty(crossrangeIdxStop), crossrangeIdxStop = length(azimuthDist); end

    for i= rangeIdx(1):rangeIdx(2)
        
        % 1. Calcul théorique de l'ouverture nécessaire (Physique OK)
        R = c * fastTime(i) / 2;
        lsynth = (c/fc) * R / (2*crossRangeResolution);
        
        % Conversion en nombre d'échantillons
        lsar = round(lsynth / (speed/prf));
        lsar = lsar + mod(lsar,2); % Impair
        
        % Fenêtre de pondération
        hn = hanning(lsar).';
        
        for j= crossrangeIdxStart:crossrangeIdxStop 
            posx = azimuthDist(j);
            posy = R;
            count = 0;
            
            % Calcul des bornes théoriques
            k_start = round(j - lsar/2 + 1);
            k_end   = round(j + lsar/2);
            
            % 2. Protection contre la réalité (Indices hors bornes)
            for k = k_start : k_end
                
                % EST-CE QUE LE RADAR ÉTAIT LÀ ?
                if k >= 1 && k <= numPulses
                    
                    % Oui, on calcule
                    td = sqrt((azimuthDist(k) - posx)^2 + posy^2) * 2/c;
                    cell = round(td * fs) + 1;
                    
                    % Est-ce que l'écho est dans la fenêtre d'écoute ?
                    if cell >= 1 && cell <= size(sigdata, 1)
                        signal = sigdata(cell, k);
                        
                        % Index correct pour la fenêtre de Hanning
                        h_idx = k - k_start + 1;
                        
                        count = count + hn(h_idx) * signal * exp(1j*2*pi*fc*(td));
                    end
                end
            end
            
            % Processed data at each of range and cross-range indices
            data(i,j)= count;
        end
        
    end
end
