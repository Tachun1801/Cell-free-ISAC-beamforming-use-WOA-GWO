function [F_star, SINR_opt] = bisection_SINR_WOA_GWO(low, high, tol, objective_func)
% bisection_SINR_WOA_GWO - Bisection search for maximum achievable SINR using WOA-GWO
%
% This function performs bisection search to find the maximum achievable
% minimum SINR for the communication beamforming optimization problem.
% Adapted from bisection_SINR_WOA to work with WOA-GWO optimizer.
%
% Inputs:
%   low - Lower bound of SINR search range
%   high - Upper bound of SINR search range
%   tol - Tolerance for bisection convergence
%   objective_func - Function handle: [F, feasible] = objective_func(gamma)
%                    where gamma is the SINR threshold to test
%
% Outputs:
%   F_star - Best feasible beamforming solution found
%   SINR_opt - Maximum achievable SINR

    F_star = [];
    SINR_opt = [];
    
    while (high - low) > tol
        mid = (high + low) / 2;
        [F, feasible] = objective_func(mid);
        
        if feasible
            low = mid;
            F_star = F;
            SINR_opt = mid;
        else
            high = mid;
            if isempty(F_star) % If no solution found yet, keep current F
                F_star = F;
            end
        end
    end
end
