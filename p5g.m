%% 5G Massive MIMO Simulation with Hybrid Beamforming & Multiple Access Techniques

clc; clear; close all;

%% System Parameters
fc = 28e9; % Carrier frequency (28 GHz)
bw = 100e6; % Bandwidth (100 MHz)
numUsers = 4; % Number of users
numTx = 64; % Number of transmit antennas (Massive MIMO)
numRx = 1; % Single antenna per user
numSubcarriers = 512; % OFDM subcarriers
cpLen = 64; % Cyclic Prefix length
modOrder = 4; % QPSK modulation
powerAllocation = [0.7, 0.2, 0.07, 0.03]; % Power levels for NOMA users
channelModels = {'UMi', 'UMa', 'InH', 'RMa'}; % 3GPP TR 38.901 Channel Models
noisePower = -90; % Noise power in dBm
multipleAccessSchemes = {'NOMA', 'OFDMA', 'SDMA'};

%% Compare Different Channel Models and Multiple Access Techniques
SINR_results = zeros(length(channelModels), length(multipleAccessSchemes));
BER_results = zeros(length(channelModels), length(multipleAccessSchemes));
SE_results = zeros(length(channelModels), length(multipleAccessSchemes));

for c = 1:length(channelModels)
    channelModel = channelModels{c};
    
    for m = 1:length(multipleAccessSchemes)
        accessScheme = multipleAccessSchemes{m};
        
        %% Generate 3GPP TR 38.901 Channel
        H = generate_channel(numTx, numUsers, channelModel, fc);
        
        %% Generate Random Data Symbols for Users
        dataSymbols = (randi([0 1], numUsers, numSubcarriers) * 2 - 1) + ...
                      1j * (randi([0 1], numUsers, numSubcarriers) * 2 - 1); % QPSK
        
        %% Apply Multiple Access Scheme
        if strcmp(accessScheme, 'NOMA')
            accessSymbols = noma_power_allocation(dataSymbols, powerAllocation);
        elseif strcmp(accessScheme, 'OFDMA')
            accessSymbols = ofdma_resource_allocation(dataSymbols, numSubcarriers);
        else % SDMA
            accessSymbols = sdma_beamforming_allocation(H, dataSymbols);
        end
        
        %% Perform OFDM Modulation
        ofdmSymbols = ofdm_modulate(accessSymbols, numSubcarriers, cpLen);
        
        %% Implement and Compare Beamforming Techniques
        numRFChains = numUsers;
        W_MMSE = hybrid_beamforming_mmse(H, numRFChains, noisePower);
        
        %% Transmit Signal Through Channel
        receivedSymbols_MMSE = signal_detection(H, W_MMSE, ofdmSymbols, cpLen);
        
        %% Compute Performance Metrics
        [~, SINR_MMSE, BER_MMSE, spectralEfficiency_MMSE] = compute_metrics(ofdmSymbols, receivedSymbols_MMSE, noisePower, modOrder, bw);
        
        %% Store Results
        SINR_results(c, m) = SINR_MMSE;
        BER_results(c, m) = BER_MMSE;
        SE_results(c, m) = spectralEfficiency_MMSE;
    end
end

%% Plot Comparison
figure;
subplot(3,1,1);
bar(SINR_results);
title('SINR Comparison Across Channel Models & Access Techniques');
set(gca, 'XTickLabel', channelModels);
legend(multipleAccessSchemes);
ylabel('SINR (dB)');
grid on;

subplot(3,1,2);
bar(BER_results);
title('BER Comparison Across Channel Models & Access Techniques');
set(gca, 'XTickLabel', channelModels);
legend(multipleAccessSchemes);
ylabel('BER');
grid on;

subplot(3,1,3);
bar(SE_results);
title('Spectral Efficiency Comparison Across Channel Models & Access Techniques');
set(gca, 'XTickLabel', channelModels);
legend(multipleAccessSchemes);
ylabel('bps/Hz');
grid on;

%% Display Results
for c = 1:length(channelModels)
    for m = 1:length(multipleAccessSchemes)
        fprintf('%s - %s -> SINR: %.2f dB, BER: %.6f, SE: %.2f bps/Hz\n', ...
                channelModels{c}, multipleAccessSchemes{m}, SINR_results(c, m), BER_results(c, m), SE_results(c, m));
    end
end

%% Function Definitions
function accessSymbols = ofdma_resource_allocation(dataSymbols, numSubcarriers)
    numUsers = size(dataSymbols, 1);
    accessSymbols = zeros(size(dataSymbols));
    for u = 1:numUsers
        accessSymbols(u, :) = dataSymbols(u, :) .* (mod(1:numSubcarriers, numUsers) == (u-1));
    end
end

function accessSymbols = sdma_beamforming_allocation(H, dataSymbols)
    [U, ~, ~] = svd(H);
    beamformingMatrix = U(:, 1:size(H,1));
    accessSymbols = beamformingMatrix' * dataSymbols;
end

function H = generate_channel(numTx, numUsers, channelModel, fc)
    % Generates 3GPP TR 38.901 channel coefficients based on the selected model

    switch channelModel
        case 'UMi' % Urban Micro
            pathLoss_dB = 32.4 + 20*log10(fc/1e9) + 20*log10(100); % Example at 100m
        case 'UMa' % Urban Macro
            pathLoss_dB = 28 + 22*log10(fc/1e9) + 20*log10(100);
        case 'InH' % Indoor Hotspot
            pathLoss_dB = 30 + 18*log10(fc/1e9) + 20*log10(50);
        case 'RMa' % Rural Macro
            pathLoss_dB = 33 + 23*log10(fc/1e9) + 20*log10(300);
        otherwise
            error('Invalid channel model selected.');
    end

    % Convert path loss to linear scale
    pathLoss = 10^(-pathLoss_dB/10);

    % Generate Rayleigh fading channel matrix
    H = (randn(numUsers, numTx) + 1j*randn(numUsers, numTx)) / sqrt(2);

    % Apply path loss
    H = H * sqrt(pathLoss);
end

function nomaSymbols = noma_power_allocation(userSymbols, powerAllocation)
    % Applies power allocation for NOMA users
    numUsers = length(powerAllocation);
    nomaSymbols = zeros(size(userSymbols));
    
    for u = 1:numUsers
        nomaSymbols(u, :) = sqrt(powerAllocation(u)) * userSymbols(u, :);
    end
end

function ofdmSymbols = ofdm_modulate(dataSymbols, numSubcarriers, cpLen)
    % Applies OFDM modulation with IFFT and Cyclic Prefix
    numUsers = size(dataSymbols, 1);
    
    % Apply IFFT across each row (user)
    timeDomainSymbols = ifft(dataSymbols, numSubcarriers, 2);
    
    % Add Cyclic Prefix
    ofdmSymbols = [timeDomainSymbols(:, end-cpLen+1:end), timeDomainSymbols];
end

function W = hybrid_beamforming_mmse(H, numRFChains, noisePower)
    % Get matrix dimensions
    [numUsers, numTx] = size(H);
    
    % Perform SVD on H (not H') to get proper dimensions
    [U, S, V] = svd(H);
    
    % Extract analog beamforming matrix with correct dimensions
    analogBF = V(:, 1:numRFChains);
    
    % Convert noise power from dBm to linear scale
    noiseLinear = 10^(noisePower/10);
    
    % Create effective channel matrix after analog beamforming
    HHH = H * analogBF;
    
    % MMSE digital beamforming
    noiseMatrix = noiseLinear * eye(numUsers);
    digitalBF = (HHH' * HHH + noiseMatrix) \ HHH';
    
    % Final hybrid beamforming matrix
    W = analogBF * digitalBF;
end

function receivedSymbols = signal_detection(H, W, ofdmSymbols, cpLen)
    % Remove cyclic prefix
    timeDomainSymbols = ofdmSymbols(:, cpLen+1:end);
    
    % Apply beamforming and channel effects
    % For each user, apply the channel and beamforming matrices
    receivedSymbols = H * W * timeDomainSymbols;
    
    % Optional: Convert back to frequency domain if needed
    % receivedSymbols = fft(receivedSymbols, [], 2);
end

function [PAPR, SINR, BER, spectralEfficiency] = compute_metrics(transmittedSymbols, receivedSymbols, noisePower, modOrder, bw)
    % Calculate Peak-to-Average Power Ratio (PAPR)
    PAPR = max(abs(transmittedSymbols(:)).^2) / mean(abs(transmittedSymbols(:)).^2);
    
    % Calculate Signal-to-Interference-plus-Noise Ratio (SINR)
    signalPower = mean(abs(receivedSymbols(:)).^2);
    noiseLinear = 10^(noisePower/10);
    SINR = 10 * log10(signalPower / noiseLinear);
    
    % Calculate Bit Error Rate (BER)
    % Ensure arrays have compatible sizes by reshaping or truncating
    cpLen = size(transmittedSymbols, 2) - size(receivedSymbols, 2);
    if cpLen > 0
        transmittedSymbolsNoCp = transmittedSymbols(:, cpLen+1:end);
    else
        transmittedSymbolsNoCp = transmittedSymbols;
    end
    
    % Ensure both arrays have the same size for comparison
    minRows = min(size(transmittedSymbolsNoCp, 1), size(receivedSymbols, 1));
    minCols = min(size(transmittedSymbolsNoCp, 2), size(receivedSymbols, 2));
    
    errors = sum(sum(sign(real(receivedSymbols(1:minRows, 1:minCols))) ~= sign(real(transmittedSymbolsNoCp(1:minRows, 1:minCols)))));
    totalBits = minRows * minCols;
    BER = errors / totalBits; % This is now a scalar
    
    % Calculate Spectral Efficiency (bps/Hz)
    spectralEfficiency = log2(modOrder) * (1 - BER);
end