%% WOA-GWO Hybrid Optimizer - Core Algorithm
% This file implements a hybrid optimization algorithm combining:
% - Whale Optimization Algorithm (WOA) for global exploration (60% iterations)
% - Grey Wolf Optimizer (GWO) for local exploitation (40% iterations)
%
% Following the pattern of WOA repo for modular code structure

function [best_position, best_fitness, convergence] = WOA_GWO_optimizer(objective_func, dim, lb, ub, max_iter, search_agents)
    % WOA_GWO_optimizer - Core hybrid WOA-GWO optimization algorithm
    %
    % Inputs:
    %   objective_func - Function handle to minimize (returns scalar fitness)
    %   dim - Dimension of the problem
    %   lb - Lower bound (scalar or 1xdim vector)
    %   ub - Upper bound (scalar or 1xdim vector)
    %   max_iter - Maximum number of iterations
    %   search_agents - Number of search agents (whales/wolves)
    %
    % Outputs:
    %   best_position - Best solution found (1 x dim)
    %   best_fitness - Best fitness value (scalar)
    %   convergence - Convergence curve (1 x max_iter)

    % Handle scalar bounds
    if isscalar(lb)
        lb = lb * ones(1, dim);
    end
    if isscalar(ub)
        ub = ub * ones(1, dim);
    end
    
    % Initialize population within bounds
    positions = zeros(search_agents, dim);
    for i = 1:search_agents
        positions(i, :) = lb + (ub - lb) .* rand(1, dim);
    end
    
    % Evaluate initial fitness (we minimize, so lower is better)
    fitness = zeros(search_agents, 1);
    for i = 1:search_agents
        fitness(i) = objective_func(positions(i, :));
    end
    
    % Find best solution (minimum for minimization)
    [best_fitness, best_idx] = min(fitness);
    best_position = positions(best_idx, :);
    
    % Track alpha, beta, delta for GWO (3 best wolves)
    [sorted_fitness, sorted_idx] = sort(fitness, 'ascend');
    alpha_pos = positions(sorted_idx(1), :);
    beta_pos = positions(sorted_idx(min(2, search_agents)), :);
    delta_pos = positions(sorted_idx(min(3, search_agents)), :);
    alpha_score = sorted_fitness(1);
    beta_score = sorted_fitness(min(2, search_agents));
    delta_score = sorted_fitness(min(3, search_agents));
    
    % Convergence tracking
    convergence = zeros(1, max_iter);
    
    % WOA-GWO iterations
    woa_ratio = 0.6; % 60% WOA, 40% GWO
    switch_iter = round(max_iter * woa_ratio);
    b = 1; % Spiral constant for WOA
    
    for iter = 1:max_iter
        a = 2 - iter * (2 / max_iter); % a decreases linearly from 2 to 0
        a2 = -1 + iter * (-1 / max_iter); % a2 decreases from -1 to -2
        
        for i = 1:search_agents
            if iter <= switch_iter
                %% WOA Phase (Exploration - 60% iterations)
                r1 = rand();
                r2 = rand();
                A = 2 * a * r1 - a;
                C = 2 * r2;
                p = rand();
                l = (a2 - 1) * rand() + 1;
                
                if p < 0.5
                    if abs(A) >= 1
                        % Exploration: random whale
                        rand_idx = randi(search_agents);
                        X_rand = positions(rand_idx, :);
                        D = abs(C * X_rand - positions(i, :));
                        positions(i, :) = X_rand - A * D;
                    else
                        % Exploitation: encircling prey
                        D = abs(C * best_position - positions(i, :));
                        positions(i, :) = best_position - A * D;
                    end
                else
                    % Spiral update
                    D_prime = abs(best_position - positions(i, :));
                    positions(i, :) = D_prime .* exp(b * l) .* cos(2 * pi * l) + best_position;
                end
            else
                %% GWO Phase (Exploitation - 40% iterations)
                r1 = rand(1, dim); r2 = rand(1, dim);
                A1 = 2 * a * r1 - a; C1 = 2 * r2;
                D_alpha = abs(C1 .* alpha_pos - positions(i, :));
                X1 = alpha_pos - A1 .* D_alpha;
                
                r1 = rand(1, dim); r2 = rand(1, dim);
                A2 = 2 * a * r1 - a; C2 = 2 * r2;
                D_beta = abs(C2 .* beta_pos - positions(i, :));
                X2 = beta_pos - A2 .* D_beta;
                
                r1 = rand(1, dim); r2 = rand(1, dim);
                A3 = 2 * a * r1 - a; C3 = 2 * r2;
                D_delta = abs(C3 .* delta_pos - positions(i, :));
                X3 = delta_pos - A3 .* D_delta;
                
                % Average
                positions(i, :) = (X1 + X2 + X3) / 3;
            end
            
            % Boundary checking
            positions(i, :) = max(min(positions(i, :), ub), lb);
            
            % Evaluate fitness
            fitness(i) = objective_func(positions(i, :));
        end
        
        % Update best solution (for WOA)
        [current_best_fitness, current_best_idx] = min(fitness);
        if current_best_fitness < best_fitness
            best_fitness = current_best_fitness;
            best_position = positions(current_best_idx, :);
        end
        
        % Update alpha, beta, delta (for GWO) - minimization
        for i = 1:search_agents
            if fitness(i) < alpha_score
                delta_score = beta_score;
                delta_pos = beta_pos;
                beta_score = alpha_score;
                beta_pos = alpha_pos;
                alpha_score = fitness(i);
                alpha_pos = positions(i, :);
            elseif fitness(i) < beta_score
                delta_score = beta_score;
                delta_pos = beta_pos;
                beta_score = fitness(i);
                beta_pos = positions(i, :);
            elseif fitness(i) < delta_score
                delta_score = fitness(i);
                delta_pos = positions(i, :);
            end
        end
        
        % Update best position to alpha (best wolf)
        if alpha_score < best_fitness
            best_fitness = alpha_score;
            best_position = alpha_pos;
        end
        
        % Store convergence
        convergence(iter) = best_fitness;
    end
end
