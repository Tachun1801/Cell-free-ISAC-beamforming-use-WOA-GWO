%% Parameters
params.N_t = 16; % Number of antennas per AP
params.M_t = 2; % Number of APs
params.U = 5; % Number of users
params.T = 1; % Number of targets

params.P = 1; % Power per AP (W)
params.P_comm_ratio = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]; % Power ratio for communications % [0.5];

% WOA-GWO Parameters
params.woa_gwo.n_whales = 30; % Number of whales/agents in WOA
params.woa_gwo.n_wolves = 20; % Number of wolves in GWO (used for hybrid selection)
params.woa_gwo.max_iter = 100; % Maximum number of iterations
params.woa_gwo.lb = -10; % Lower bound for optimization variables
params.woa_gwo.ub = 10; % Upper bound for optimization variables
params.woa_gwo.sensing_weight = 0.01; % Weight for sensing SNR in fitness function
params.woa_gwo.penalty_factor = 100; % Penalty factor for constraint violations
params.woa_gwo.feasibility_tolerance = 0.9; % Tolerance factor for feasibility check

% Bisection search parameters for SINR optimization
params.bisect.low = 0.01;    % Lower bound for SINR search
params.bisect.high = 100;    % Upper bound for SINR search
params.bisect.tol = 0.01;    % Tolerance for bisection convergence

% Noise
params.sigmasq_ue = 1; % UE receiver noise
params.sigmasq_radar_rcs = 0.1; % Radar RCS variable 
params.sigmasq_radar_receiver = 1; % Radar receiver noise

params.repetitions = 1000;

% Geometry setup
params.geo.line_length = 100;
params.geo.UE_y = 50;
params.geo.target_y = 50;
params.geo.min_dist = 0;
params.geo.max_dist = params.geo.line_length;