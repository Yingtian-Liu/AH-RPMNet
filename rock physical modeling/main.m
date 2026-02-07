clc;
clear;
close all;
%% Load data
true_vp = 5452;
true_vs = 3145;
rho = 2.648;
por = 0.01109;
clay = 0.664007; 
sand = 0.205918;
plagioclase = 0.026444;
siderite = 0.092541;
water_saturation = 1; 
%% Basic parameters
k = [37; 75.6; 123.7]; % Bulk moduli of quartz, plagioclase, and siderite (GPa)
mu = [44; 25.6; 51]; % Shear moduli of quartz, plagioclase, and siderite (GPa)
k_clay = 25; % Bulk modulus of clay (GPa)
mu_clay = 9; % Shear modulus of clay (GPa)
k_pore = 0.04; % Pore bulk modulus (GPa)
mu_pore = 0; % Pore shear modulus (GPa)

%% Brittle minerals (VRH)
fprintf('Starting VRH mixing for brittle frame moduli (quartz, plagioclase, siderite)...\n');
total_brittle = sand + plagioclase + siderite;
f_quartz = sand / total_brittle;  
f_plagioclase = plagioclase / total_brittle; 
f_siderite = siderite / total_brittle; 

% Voigt upper bound (uniform strain assumption)
k_voigt = f_quartz * k(1) + f_plagioclase * k(2) + f_siderite * k(3);
mu_voigt = f_quartz * mu(1) + f_plagioclase * mu(2) + f_siderite * mu(3);

% Reuss lower bound (uniform stress assumption)
k_reuss = 1 / (f_quartz/k(1) + f_plagioclase/k(2) + f_siderite/k(3));
mu_reuss = 1 / (f_quartz/mu(1) + f_plagioclase/mu(2) + f_siderite/mu(3));

% Hill average (VRH average)
k_brittle = (k_voigt + k_reuss) / 2;
mu_brittle = (mu_voigt + mu_reuss) / 2;
fprintf('Brittle frame moduli calculation completed!\n');

%% Solid matrix (DEM)
asp = 1.0; phic = 0.46;   
fprintf('Starting DEM mixing for solid matrix moduli (clay as background)...\n');
total_solid = sand + plagioclase + siderite + clay;
f_brittle = (sand + plagioclase + siderite) / total_solid;
f_clay = clay / total_solid;

[k_dem, mu_dem, frac_temp] = dem(k_clay, mu_clay, k_brittle, mu_brittle, asp, phic);
[~, idx] = min(abs(frac_temp - f_brittle));
k_matrix = k_dem(idx);
mu_matrix = mu_dem(idx);
brittle_fraction_dem = frac_temp(idx);

%% Adding pores using SCA to build dry rock frame
asp_pore = 0.83; 
fprintf('Starting SCA mixing for dry rock moduli (adding pores)...\n');
k_sca = [k_matrix; k_pore];
mu_sca = [mu_matrix; mu_pore];
asp_sca = [0.96; asp_pore]; 
x_sca = [1 - por; por];
[k_dry, mu_dry] = SCA(k_sca, mu_sca, asp_sca, x_sca);

%% Calculating fluid bulk modulus using Domenico (water saturated)
k_fl = 2.25; 
rho_water = 1.01; 
fprintf('Starting Gassmann fluid substitution (adding water with given saturation)...\n');
k_sat_water = gassmnk(k_dry, 0, k_fl, k_matrix, por);
mu_sat_water = mu_dry;
fprintf('Water-saturated rock moduli calculation completed!\n');

%% Calculating velocity model
pre_vp = 1e3*sqrt((k_sat_water+(4/3)*mu_sat_water)./rho);
pre_vs = 1e3*sqrt(mu_sat_water./rho);
fprintf('Predicted P-wave velocity = %.0f m/s\n', pre_vp);
fprintf('Predicted P-wave velocity error = %.2f%%\n', abs(pre_vp-true_vp)/true_vp*100);
fprintf('Predicted S-wave velocity = %.0f m/s\n', pre_vs);
fprintf('Predicted S-wave velocity error = %.2f%%\n', abs(pre_vs-true_vs)/true_vs*100);
fprintf('Total predicted error = %.2f%%\n', abs(pre_vp-true_vp)/true_vp*100 + abs(pre_vs-true_vs)/true_vs*100);

