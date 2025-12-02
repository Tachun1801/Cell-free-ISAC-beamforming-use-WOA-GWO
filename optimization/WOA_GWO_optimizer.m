%% WOA-GWO Hybrid Optimizer for Cell-Free ISAC Beamforming
% This file implements a hybrid optimization algorithm combining:
% - Whale Optimization Algorithm (WOA) for global exploration
% - Grey Wolf Optimizer (GWO) for local exploitation

function [F_star, min_SINR, sensing_SNR] = WOA_GWO_optimizer(H_comm, sigmasq_ue, P_comm, F_sensing, sensing_beamsteering, sigmasq_radar_rcs, params)
    % WOA_GWO_optimizer - Hybrid WOA-GWO optimization for beamforming
    %
    % Inputs:
    %   H_comm - Communication channel matrix (U x M x N)
    %   sigmasq_ue - UE receiver noise variance
    %   P_comm - Communication power budget per AP
    %   F_sensing - Sensing beamforming matrix (T x M x N)
    %   sensing_beamsteering - Sensing beamsteering vectors
    %   sigmasq_radar_rcs - Radar RCS variance
    %   params - Structure containing WOA-GWO parameters
    %
    % Outputs:
    %   F_star - Optimized communication beamforming matrix (U x M x N)
    %   min_SINR - Minimum SINR achieved
    %   sensing_SNR - Sensing SNR achieved

    [U, M, N] = size(H_comm);
    dim = U * M * N * 2; % Dimension (real + imag parts for each beamforming element)
    
    % Get WOA-GWO parameters
    n_agents = params.woa_gwo.n_whales;
    max_iter = params.woa_gwo.max_iter;
    lb = params.woa_gwo.lb;
    ub = params.woa_gwo.ub;
    
    % Initialize population
    population = lb + (ub - lb) * rand(n_agents, dim);
    
    % Evaluate initial fitness
    fitness = zeros(n_agents, 1);
    for i = 1:n_agents
        F_test = decode_solution(population(i, :), U, M, N);
        F_test = normalize_power(F_test, P_comm, M);
        fitness(i) = compute_fitness(H_comm, F_test, F_sensing, sigmasq_ue, sensing_beamsteering, sigmasq_radar_rcs);
    end
    
    % Find best solution
    [best_fitness, best_idx] = max(fitness);
    best_position = population(best_idx, :);
    
    % Track best for GWO
    [sorted_fitness, sorted_idx] = sort(fitness, 'descend');
    alpha_pos = population(sorted_idx(1), :);
    beta_pos = population(sorted_idx(min(2, n_agents)), :);
    delta_pos = population(sorted_idx(min(3, n_agents)), :);
    alpha_score = sorted_fitness(1);
    beta_score = sorted_fitness(min(2, n_agents));
    delta_score = sorted_fitness(min(3, n_agents));
    
    % WOA-GWO iterations
    woa_ratio = 0.6; % 60% WOA, 40% GWO
    switch_iter = round(max_iter * woa_ratio);
    
    for iter = 1:max_iter
        a = 2 - iter * (2 / max_iter); % a decreases linearly from 2 to 0
        a2 = -1 + iter * (-1 / max_iter); % a2 decreases from -1 to -2
        
        for i = 1:n_agents
            if iter <= switch_iter
                % WOA Phase (Exploration)
                population(i, :) = WOA_update(population(i, :), best_position, a, a2, dim, lb, ub, population, n_agents);
            else
                % GWO Phase (Exploitation)
                population(i, :) = GWO_update(population(i, :), alpha_pos, beta_pos, delta_pos, a, dim, lb, ub);
            end
            
            % Evaluate fitness
            F_test = decode_solution(population(i, :), U, M, N);
            F_test = normalize_power(F_test, P_comm, M);
            fitness(i) = compute_fitness(H_comm, F_test, F_sensing, sigmasq_ue, sensing_beamsteering, sigmasq_radar_rcs);
        end
        
        % Update best solution (for WOA)
        [current_best_fitness, current_best_idx] = max(fitness);
        if current_best_fitness > best_fitness
            best_fitness = current_best_fitness;
            best_position = population(current_best_idx, :);
        end
        
        % Update alpha, beta, delta (for GWO)
        for i = 1:n_agents
            if fitness(i) > alpha_score
                delta_score = beta_score;
                delta_pos = beta_pos;
                beta_score = alpha_score;
                beta_pos = alpha_pos;
                alpha_score = fitness(i);
                alpha_pos = population(i, :);
            elseif fitness(i) > beta_score
                delta_score = beta_score;
                delta_pos = beta_pos;
                beta_score = fitness(i);
                beta_pos = population(i, :);
            elseif fitness(i) > delta_score
                delta_score = fitness(i);
                delta_pos = population(i, :);
            end
        end
        
        % Update best position to alpha (best wolf)
        if alpha_score > best_fitness
            best_fitness = alpha_score;
            best_position = alpha_pos;
        end
    end
    
    % Decode final solution
    F_star = decode_solution(best_position, U, M, N);
    F_star = normalize_power(F_star, P_comm, M);
    
    % Compute final metrics
    SINR = compute_SINR(H_comm, F_star, F_sensing, sigmasq_ue);
    min_SINR = min(SINR);
    sensing_SNR = compute_sensing_SNR(sigmasq_radar_rcs, sensing_beamsteering, F_star, F_sensing);
end

%% WOA Update Function
function new_pos = WOA_update(pos, best_pos, a, a2, dim, lb, ub, population, n_agents)
    r1 = rand();
    r2 = rand();
    A = 2 * a * r1 - a; % Eq. (2.3)
    C = 2 * r2; % Eq. (2.4)
    b = 1; % Spiral constant
    l = (a2 - 1) * rand() + 1; % Eq. (2.5)
    p = rand();
    
    if p < 0.5
        if abs(A) >= 1
            % Exploration: Search for prey (random whale)
            rand_idx = randi(n_agents);
            X_rand = population(rand_idx, :);
            D = abs(C * X_rand - pos); % Eq. (2.7)
            new_pos = X_rand - A * D; % Eq. (2.8)
        else
            % Exploitation: Encircling prey
            D = abs(C * best_pos - pos); % Eq. (2.1)
            new_pos = best_pos - A * D; % Eq. (2.2)
        end
    else
        % Spiral updating position
        D_prime = abs(best_pos - pos); % Eq. (2.5)
        new_pos = D_prime .* exp(b * l) .* cos(2 * pi * l) + best_pos; % Eq. (2.5)
    end
    
    % Boundary check
    new_pos = max(min(new_pos, ub), lb);
end

%% GWO Update Function
function new_pos = GWO_update(pos, alpha_pos, beta_pos, delta_pos, a, dim, lb, ub)
    % Calculate coefficients
    r1 = rand(1, dim);
    r2 = rand(1, dim);
    A1 = 2 * a * r1 - a;
    C1 = 2 * r2;
    
    r1 = rand(1, dim);
    r2 = rand(1, dim);
    A2 = 2 * a * r1 - a;
    C2 = 2 * r2;
    
    r1 = rand(1, dim);
    r2 = rand(1, dim);
    A3 = 2 * a * r1 - a;
    C3 = 2 * r2;
    
    % Calculate D vectors
    D_alpha = abs(C1 .* alpha_pos - pos);
    D_beta = abs(C2 .* beta_pos - pos);
    D_delta = abs(C3 .* delta_pos - pos);
    
    % Calculate X vectors
    X1 = alpha_pos - A1 .* D_alpha;
    X2 = beta_pos - A2 .* D_beta;
    X3 = delta_pos - A3 .* D_delta;
    
    % New position is average of X1, X2, X3
    new_pos = (X1 + X2 + X3) / 3;
    
    % Boundary check
    new_pos = max(min(new_pos, ub), lb);
end

%% Decode Solution: Convert flat vector to beamforming matrix
function F = decode_solution(x, U, M, N)
    total_elements = U * M * N;
    real_part = reshape(x(1:total_elements), [U, M, N]);
    imag_part = reshape(x(total_elements+1:end), [U, M, N]);
    F = real_part + 1j * imag_part;
end

%% Normalize Power: Ensure power constraint per AP is satisfied
function F_norm = normalize_power(F, P_max, M)
    [U, ~, N] = size(F);
    F_norm = F;
    
    for m = 1:M
        % Compute total power at AP m
        power_m = sum(sum(abs(F(:, m, :)).^2, 3), 1);
        
        if power_m > P_max
            % Scale down to meet power constraint
            scale_factor = sqrt(P_max / power_m);
            F_norm(:, m, :) = F(:, m, :) * scale_factor;
        end
    end
end

%% Fitness Function: Maximize minimum SINR with sensing consideration
function fitness = compute_fitness(H_comm, F_comm, F_sensing, sigmasq_ue, sensing_beamsteering, sigmasq_radar_rcs)
    % Compute SINR for all users
    SINR = compute_SINR(H_comm, F_comm, F_sensing, sigmasq_ue);
    min_SINR = min(SINR);
    
    % Compute sensing SNR
    SSNR = compute_sensing_SNR(sigmasq_radar_rcs, sensing_beamsteering, F_comm, F_sensing);
    
    % Fitness is the minimum SINR (we maximize this)
    % Add a small term for sensing to encourage good sensing performance
    fitness = min_SINR + 0.01 * SSNR;
    
    % Penalize negative or very low SINR
    if min_SINR < 0
        fitness = fitness - 100;
    end
end

%% JSC Optimizer using WOA-GWO
function [F_comm, F_sensing_opt, SSNR_opt, feasible] = WOA_GWO_JSC_optimizer(H_comm, sigmasq_ue, gamma_min, sensing_beamsteering, sens_streams, sigmasq_radar_rcs, P_all, params)
    % WOA_GWO_JSC_optimizer - Joint Sensing-Communication optimization
    %
    % Inputs:
    %   H_comm - Communication channel matrix (U x M x N)
    %   sigmasq_ue - UE receiver noise variance
    %   gamma_min - Minimum SINR threshold
    %   sensing_beamsteering - Sensing beamsteering vectors
    %   sens_streams - Number of sensing streams
    %   sigmasq_radar_rcs - Radar RCS variance
    %   P_all - Total power budget per AP
    %   params - Structure containing WOA-GWO parameters
    %
    % Outputs:
    %   F_comm - Optimized communication beamforming matrix (U x M x N)
    %   F_sensing_opt - Optimized sensing beamforming matrix
    %   SSNR_opt - Optimal sensing SNR
    %   feasible - Feasibility flag

    [U, M, N] = size(H_comm);
    num_streams = U + sens_streams;
    dim = num_streams * M * N * 2; % Dimension for all streams
    
    % Get WOA-GWO parameters
    n_agents = params.woa_gwo.n_whales;
    max_iter = params.woa_gwo.max_iter;
    lb = params.woa_gwo.lb;
    ub = params.woa_gwo.ub;
    
    % Initialize population
    population = lb + (ub - lb) * rand(n_agents, dim);
    
    % Evaluate initial fitness
    fitness = zeros(n_agents, 1);
    for i = 1:n_agents
        [F_all_test, ~, ~] = decode_jsc_solution(population(i, :), U, M, N, sens_streams);
        F_all_test = normalize_power_jsc(F_all_test, P_all, M, num_streams);
        fitness(i) = compute_jsc_fitness(H_comm, F_all_test, U, sens_streams, sigmasq_ue, sensing_beamsteering, sigmasq_radar_rcs, gamma_min);
    end
    
    % Find best solution
    [best_fitness, best_idx] = max(fitness);
    best_position = population(best_idx, :);
    
    % Track best for GWO
    [sorted_fitness, sorted_idx] = sort(fitness, 'descend');
    alpha_pos = population(sorted_idx(1), :);
    beta_pos = population(sorted_idx(min(2, n_agents)), :);
    delta_pos = population(sorted_idx(min(3, n_agents)), :);
    alpha_score = sorted_fitness(1);
    beta_score = sorted_fitness(min(2, n_agents));
    delta_score = sorted_fitness(min(3, n_agents));
    
    % WOA-GWO iterations
    woa_ratio = 0.6;
    switch_iter = round(max_iter * woa_ratio);
    
    for iter = 1:max_iter
        a = 2 - iter * (2 / max_iter);
        a2 = -1 + iter * (-1 / max_iter);
        
        for i = 1:n_agents
            if iter <= switch_iter
                population(i, :) = WOA_update(population(i, :), best_position, a, a2, dim, lb, ub, population, n_agents);
            else
                population(i, :) = GWO_update(population(i, :), alpha_pos, beta_pos, delta_pos, a, dim, lb, ub);
            end
            
            % Evaluate fitness
            [F_all_test, ~, ~] = decode_jsc_solution(population(i, :), U, M, N, sens_streams);
            F_all_test = normalize_power_jsc(F_all_test, P_all, M, num_streams);
            fitness(i) = compute_jsc_fitness(H_comm, F_all_test, U, sens_streams, sigmasq_ue, sensing_beamsteering, sigmasq_radar_rcs, gamma_min);
        end
        
        % Update best solution
        [current_best_fitness, current_best_idx] = max(fitness);
        if current_best_fitness > best_fitness
            best_fitness = current_best_fitness;
            best_position = population(current_best_idx, :);
        end
        
        % Update alpha, beta, delta
        for i = 1:n_agents
            if fitness(i) > alpha_score
                delta_score = beta_score;
                delta_pos = beta_pos;
                beta_score = alpha_score;
                beta_pos = alpha_pos;
                alpha_score = fitness(i);
                alpha_pos = population(i, :);
            elseif fitness(i) > beta_score
                delta_score = beta_score;
                delta_pos = beta_pos;
                beta_score = fitness(i);
                beta_pos = population(i, :);
            elseif fitness(i) > delta_score
                delta_score = fitness(i);
                delta_pos = population(i, :);
            end
        end
        
        if alpha_score > best_fitness
            best_fitness = alpha_score;
            best_position = alpha_pos;
        end
    end
    
    % Decode final solution
    [F_all_opt, F_comm, F_sensing_opt] = decode_jsc_solution(best_position, U, M, N, sens_streams);
    F_all_opt = normalize_power_jsc(F_all_opt, P_all, M, num_streams);
    F_comm = F_all_opt(1:U, :, :);
    if sens_streams > 0
        F_sensing_opt = F_all_opt(U+1:end, :, :);
    else
        F_sensing_opt = zeros(1, M, N);
    end
    
    % Compute final metrics
    SINR = compute_SINR(H_comm, F_comm, F_sensing_opt, sigmasq_ue);
    min_SINR = min(SINR);
    SSNR_opt = compute_sensing_SNR(sigmasq_radar_rcs, sensing_beamsteering, F_comm, F_sensing_opt);
    
    % Check feasibility
    feasible = (min_SINR >= gamma_min * 0.9); % Allow 10% tolerance
end

%% Decode JSC Solution
function [F_all, F_comm, F_sensing] = decode_jsc_solution(x, U, M, N, sens_streams)
    num_streams = U + sens_streams;
    total_elements = num_streams * M * N;
    real_part = reshape(x(1:total_elements), [num_streams, M, N]);
    imag_part = reshape(x(total_elements+1:end), [num_streams, M, N]);
    F_all = real_part + 1j * imag_part;
    F_comm = F_all(1:U, :, :);
    if sens_streams > 0
        F_sensing = F_all(U+1:end, :, :);
    else
        F_sensing = zeros(1, M, N);
    end
end

%% Normalize Power for JSC
function F_norm = normalize_power_jsc(F, P_max, M, num_streams)
    [~, ~, N] = size(F);
    F_norm = F;
    
    for m = 1:M
        % Compute total power at AP m across all streams
        power_m = sum(sum(abs(F(:, m, :)).^2, 3), 1);
        
        if power_m > P_max
            scale_factor = sqrt(P_max / power_m);
            F_norm(:, m, :) = F(:, m, :) * scale_factor;
        end
    end
end

%% JSC Fitness Function
function fitness = compute_jsc_fitness(H_comm, F_all, U, sens_streams, sigmasq_ue, sensing_beamsteering, sigmasq_radar_rcs, gamma_min)
    F_comm = F_all(1:U, :, :);
    if sens_streams > 0
        F_sensing = F_all(U+1:end, :, :);
    else
        F_sensing = zeros(1, size(F_all, 2), size(F_all, 3));
    end
    
    % Compute SINR for all users
    SINR = compute_SINR(H_comm, F_comm, F_sensing, sigmasq_ue);
    min_SINR = min(SINR);
    
    % Compute sensing SNR
    SSNR = compute_sensing_SNR(sigmasq_radar_rcs, sensing_beamsteering, F_comm, F_sensing);
    
    % Penalize if minimum SINR constraint is violated
    if min_SINR < gamma_min
        penalty = 100 * (gamma_min - min_SINR);
    else
        penalty = 0;
    end
    
    % Maximize sensing SNR while maintaining SINR constraint
    fitness = SSNR - penalty;
end
