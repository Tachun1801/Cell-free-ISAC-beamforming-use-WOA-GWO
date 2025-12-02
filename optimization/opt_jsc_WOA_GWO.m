function [F_comm, F_sensing, feasible, SSNR_opt] = opt_jsc_WOA_GWO(H_comm, sigmasq_comm, gamma, sensing_beamsteering, sens_streams, sigmasq_sens, P_all)
% opt_jsc_WOA_GWO - WOA-GWO based Joint Sensing-Communication Optimization
%
% This function performs joint optimization of communication and sensing
% beamforming vectors using the hybrid WOA-GWO algorithm to maximize
% sensing SNR while maintaining SINR constraints.
%
% Inputs:
%   H_comm - Communication channel matrix (U x M x N)
%   sigmasq_comm - UE receiver noise variance
%   gamma - Minimum SINR threshold to maintain
%   sensing_beamsteering - Sensing beamsteering vectors (T x M x N)
%   sens_streams - Number of sensing streams
%   sigmasq_sens - Radar RCS variance
%   P_all - Total power budget per AP
%
% Outputs:
%   F_comm - Optimized communication beamforming matrix (U x M x N)
%   F_sensing - Optimized sensing beamforming matrix
%   feasible - Boolean indicating if SINR constraints are satisfied
%   SSNR_opt - Optimal sensing SNR achieved

    [U, M, N] = size(H_comm);
    num_streams = U + sens_streams;
    dim = num_streams * M * N * 2; % Dimension for all streams (real + imag)
    
    % WOA-GWO parameters
    n_agents = 30;      % Number of search agents
    max_iter = 100;     % Maximum iterations
    lb = -10;           % Lower bound
    ub = 10;            % Upper bound
    penalty_factor = 100; % Penalty for constraint violations
    
    % Create objective function for WOA-GWO optimizer
    % We minimize (-SSNR + penalty) which maximizes SSNR with SINR constraints
    objective = @(x) compute_jsc_objective_WOA_GWO(x, H_comm, sigmasq_comm, gamma, sensing_beamsteering, sigmasq_sens, P_all, penalty_factor, U, M, N, sens_streams);
    
    % Call WOA-GWO optimizer
    [best_position, ~, ~] = WOA_GWO_optimizer(objective, dim, lb, ub, max_iter, n_agents);
    
    % Reconstruct beamforming matrices
    [F_comm, F_sensing] = reconstruct_jsc_beamforming(best_position, U, M, N, sens_streams, P_all);
    
    % Compute final metrics
    SINR = compute_SINR(H_comm, F_comm, F_sensing, sigmasq_comm);
    min_SINR = min(SINR);
    SSNR_opt = compute_SSNR(sigmasq_sens, sensing_beamsteering, F_comm, F_sensing);
    
    % Check feasibility (90% tolerance)
    feasible = (min_SINR >= gamma * 0.9);
end

%% Compute JSC Objective for WOA-GWO
function obj = compute_jsc_objective_WOA_GWO(x, H_comm, sigmasq_comm, gamma, sensing_beamsteering, sigmasq_sens, P_all, penalty_factor, U, M, N, sens_streams)
    % Reconstruct beamforming matrices
    [F_comm, F_sensing] = reconstruct_jsc_beamforming(x, U, M, N, sens_streams, P_all);
    
    % Compute SINR for all users
    SINR = compute_SINR(H_comm, F_comm, F_sensing, sigmasq_comm);
    min_SINR = min(SINR);
    
    % Compute sensing SNR
    SSNR = compute_SSNR(sigmasq_sens, sensing_beamsteering, F_comm, F_sensing);
    
    % Penalize if minimum SINR constraint is violated
    if min_SINR < gamma
        penalty = penalty_factor * (gamma - min_SINR);
    else
        penalty = 0;
    end
    
    % Objective: minimize (-SSNR + penalty)
    % This maximizes SSNR while maintaining SINR constraints
    obj = -SSNR + penalty;
end

%% Compute Sensing SNR
function SSNR = compute_SSNR(sigmasq_sens, sensing_beamsteering, F_comm, F_sensing)
    % Wrapper for compute_sensing_SNR utility function
    SSNR = compute_sensing_SNR(sigmasq_sens, sensing_beamsteering, F_comm, F_sensing);
end

%% Reconstruct JSC Beamforming Matrices
function [F_comm, F_sensing] = reconstruct_jsc_beamforming(x, U, M, N, sens_streams, P_max)
    num_streams = U + sens_streams;
    total_elements = num_streams * M * N;
    
    % Decode from flat vector
    real_part = reshape(x(1:total_elements), [num_streams, M, N]);
    imag_part = reshape(x(total_elements+1:end), [num_streams, M, N]);
    F_all = real_part + 1j * imag_part;
    
    % Normalize power per AP
    for m = 1:M
        power_m = sum(sum(abs(F_all(:, m, :)).^2, 3), 1);
        if power_m > P_max
            scale_factor = sqrt(P_max / power_m);
            F_all(:, m, :) = F_all(:, m, :) * scale_factor;
        end
    end
    
    % Split into comm and sensing
    F_comm = F_all(1:U, :, :);
    if sens_streams > 0
        F_sensing = F_all(U+1:end, :, :);
    else
        F_sensing = zeros(1, M, N);
    end
end
