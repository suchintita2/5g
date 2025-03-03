%% 5G Massive MIMO-NOMA Simulation with Hybrid Beamforming & OFDM

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
channelModel = 'UMi'; % 3GPP TR 38.901 Channel Model
noisePower = -90; % Noise power in dBm

%% Generate 3GPP TR 38.901 Channel
H = generate_channel(numTx, numUsers, channelModel, fc);

%% Generate Random Data Symbols for Users
dataSymbols = (randi([0 1], numUsers, numSubcarriers) * 2 - 1) + ...
              1j * (randi([0 1], numUsers, numSubcarriers) * 2 - 1); % QPSK

%% Apply NOMA Power Allocation
nomaSymbols = noma_power_allocation(dataSymbols, powerAllocation);

%% Perform OFDM Modulation
ofdmSymbols = ofdm_modulate(nomaSymbols, numSubcarriers, cpLen);

%% Implement Hybrid Beamforming
numRFChains = numUsers;
W = hybrid_beamforming(H, numRFChains);

%% Transmit Signal Through Channel
receivedSymbols = signal_detection(H, W, ofdmSymbols, cpLen);

%% Compute Performance Metrics
[PAPR, SINR] = compute_metrics(ofdmSymbols, receivedSymbols, noisePower);

%% Display Results
fprintf('PAPR: %.2f dB\n', 10*log10(PAPR));
fprintf('SINR: %.2f dB\n', SINR);


%% Function Definitions
function H = generate_channel(numTx, numUsers, channelModel, fc)
    switch channelModel
        case 'UMi'
            pathLoss_dB = 32.4 + 20*log10(fc/1e9) + 20*log10(100);
        case 'UMa'
            pathLoss_dB = 28 + 22*log10(fc/1e9) + 20*log10(100);
        case 'InH'
            pathLoss_dB = 30 + 18*log10(fc/1e9) + 20*log10(50);
        case 'RMa'
            pathLoss_dB = 33 + 23*log10(fc/1e9) + 20*log10(300);
    end
    pathLoss = 10^(-pathLoss_dB/10);
    H = (randn(numUsers, numTx) + 1j*randn(numUsers, numTx)) / sqrt(2);
    H = H * sqrt(pathLoss);
end

function ofdmSymbols = ofdm_modulate(dataSymbols, numSubcarriers, cpLen)
    % Apply IFFT across each row (user)
    numUsers = size(dataSymbols, 1);
    timeDomainSymbols = ifft(dataSymbols, numSubcarriers, 2);
    
    % Add cyclic prefix
    ofdmSymbols = [timeDomainSymbols(:, end-cpLen+1:end), timeDomainSymbols];
end

function nomaSymbols = noma_power_allocation(userSymbols, powerAllocation)
    numUsers = length(powerAllocation);
    nomaSymbols = zeros(size(userSymbols));
    for u = 1:numUsers
        nomaSymbols(u, :) = sqrt(powerAllocation(u)) * userSymbols(u, :);
    end
end

function W = hybrid_beamforming(H, numRFChains)
    [~, ~, V] = svd(H); % Singular Value Decomposition
    analogBF = V(:, 1:numRFChains); % Analog beamforming (Tx × RFChains)
    digitalBF = pinv(H * analogBF); % Digital beamforming (RFChains × Users)
    W = analogBF * digitalBF; % (Tx × Users)
end


function receivedSymbols = signal_detection(H, W, ofdmSymbols, cpLen)
    % Remove cyclic prefix
    transmittedSymbols = ofdmSymbols(:, cpLen+1:end);
    
    % Apply beamforming and channel effects
    receivedSymbols = H * W * transmittedSymbols;
end

function [PAPR, SINR] = compute_metrics(transmittedSymbols, receivedSymbols, noisePower)
    PAPR = max(abs(transmittedSymbols(:)).^2) / mean(abs(transmittedSymbols(:)).^2);
    signalPower = mean(abs(receivedSymbols(:)).^2);
    noise = 10^(noisePower/10);
    SINR = 10 * log10(signalPower / noise);
end
