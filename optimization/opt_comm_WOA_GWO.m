function [F_star, feasible] = opt_comm_WOA_GWO(H_comm, sigmasq_comm, P_comm, F_sensing, gamma)
% opt_comm_WOA_GWO - WOA-GWO based Communication Beamforming Optimization
%
% This function optimizes communication beamforming vectors using the
% hybrid WOA-GWO algorithm to maximize minimum SINR while satisfying
% power constraints.
%
% Inputs:
%   H_comm - Communication channel matrix (U x M x N)
%   sigmasq_comm - UE receiver noise variance
%   P_comm - Communication power budget per AP
%   F_sensing - Sensing beamforming matrix (T x M x N)
%   gamma - Target SINR threshold
%
% Outputs:
%   F_star - Optimized communication beamforming matrix (U x M x N)
%   feasible - Boolean indicating if solution meets SINR threshold

    [U, M, N] = size(H_comm);
    dim = U * M * N * 2; % Dimension (real + imag parts)
    
    % WOA-GWO parameters
    n_agents = 30;      % Number of search agents
    max_iter = 100;     % Maximum iterations
    lb = -10;           % Lower bound
    ub = 10;            % Upper bound
    penalty_factor = 100; % Penalty for constraint violations
    
    % Create objective function for WOA-GWO optimizer
    % We minimize (-min_SINR + penalty) which is equivalent to maximizing min_SINR
    objective = @(x) compute_comm_objective_WOA_GWO(x, H_comm, F_sensing, sigmasq_comm, P_comm, gamma, penalty_factor, U, M, N);
    
    % Call WOA-GWO optimizer
    [best_position, ~, ~] = WOA_GWO_optimizer(objective, dim, lb, ub, max_iter, n_agents);
    
    % Reconstruct beamforming matrix
    F_star = reconstruct_beamforming(best_position, U, M, N, P_comm);
    
    % Check feasibility
    SINR = compute_SINR(H_comm, F_star, F_sensing, sigmasq_comm);
    min_SINR = min(SINR);
    feasible = (min_SINR >= gamma * 0.9); % 90% tolerance for feasibility
end

%% Compute Communication Objective for WOA-GWO
function obj = compute_comm_objective_WOA_GWO(x, H_comm, F_sensing, sigmasq_comm, P_comm, gamma, penalty_factor, U, M, N)
    % Reconstruct beamforming matrix from flat vector
    F = reconstruct_beamforming(x, U, M, N, P_comm);
    
    % Compute SINR for all users
    SINR = compute_SINR(H_comm, F, F_sensing, sigmasq_comm);
    min_SINR = min(SINR);
    
    % Objective: minimize (-min_SINR + penalty)
    % This maximizes min_SINR while penalizing constraint violations
    if min_SINR < gamma
        penalty = penalty_factor * (gamma - min_SINR);
    else
        penalty = 0;
    end
    
    obj = -min_SINR + penalty;
end

%% Reconstruct Beamforming Matrix
function F = reconstruct_beamforming(x, U, M, N, P_max)
    % Decode from flat vector
    total_elements = U * M * N;
    real_part = reshape(x(1:total_elements), [U, M, N]);
    imag_part = reshape(x(total_elements+1:end), [U, M, N]);
    F = real_part + 1j * imag_part;
    
    % Normalize power per AP
    for m = 1:M
        power_m = sum(sum(abs(F(:, m, :)).^2, 3), 1);
        if power_m > P_max
            scale_factor = sqrt(P_max / power_m);
            F(:, m, :) = F(:, m, :) * scale_factor;
        end
    end
end
