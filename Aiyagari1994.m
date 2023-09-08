% Example based on Aiyagari (1994).
% The purpose of this program is two compare two methods for computing the
% stationary distribution in HA models.
% The first method is the default implemented in the VFI toolkit (from here
% on called RK), which is quite common and is explained e.g. in Ljiunqvist
% and Sargent textbook.
% The second method is based on Eugene Tan Econ Letters paper, called 'AD'
% The second method is significantly faster: depending on grid sizes, the
% speed up is 10 times or more. For why it is faster, see Eugene's paper.
% Moreover, the second method does not require GPU.

% NOTE: To run this program you need also the files 
% - StationaryDist_Case1
% - StationaryDist_Case1_Iteration_raw_AD
% - StationaryDist_Case1_Iteration_raw_AD_vec
% Replace the function StationaryDist_Case1 in the toolkit with the one
% provided on my github page
% NOTE: The function StationaryDist_Case1_Iteration_raw_AD is not used but
% is meant as an illustration of the algorithm (it is written with loops,
% easier to understand).

% Link to Eugene's paper:
% A Fast and Low Computational Memory Algorithm For Non-Stochastic Simulations in Heterogeneous Agent Models
% Eugene Tan, University of Toronto
% https://drive.google.com/file/d/1zm1TWy9I-wh256XngpiZaIItQN6qQCkW/view

clear
clc 
close all
% Set the path so that Matlab finds the toolkit files. This is
% user-specific, it depends on where you have the toolkit on your computer
addpath(genpath(fullfile('..','VFIToolkit-matlab-master')))
format long g

%% Set some basic variables

% VFI Toolkit thinks of there as being:
% k: an endogenous state variable (assets)
% z: an exogenous state variable (exogenous labor supply)

% Size of the grids
n_k = 512;%2^9;
n_z = 21; %21;

% Parameters
Params.beta=0.96; %Model period is one-sixth of a year
Params.alpha=0.36;
Params.delta=0.08;
Params.mu=3;
Params.sigma=0.2;
Params.rho=0.6;

%% Set up the exogenous shock process
% Create markov process for the exogenous labour productivity, l.
Tauchen_q=3; % Footnote 33 of Aiyagari(1993WP, pg 25) implicitly says that he uses q=3
[z_grid,pi_z]=discretizeAR1_Tauchen(0,Params.rho,sqrt((1-Params.rho^2)*Params.sigma^2),n_z,Tauchen_q);
% Note: sigma is standard deviations of s, input needs to be standard deviation of the innovations
% Because s is AR(1), the variance of the innovations is (1-rho^2)*sigma^2

[z_mean,z_variance,z_corr,~]=MarkovChainMoments(z_grid,pi_z);
z_grid=exp(z_grid);
% Get some info on the markov process
[Expectation_l,~,~,~]=MarkovChainMoments(z_grid,pi_z); %Since l is exogenous, this will be it's eqm value 
% Note: Aiyagari (1994) actually then normalizes l by dividing it by Expectation_l (so that the resulting process has expectation equal to 1)
z_grid=z_grid./Expectation_l;
[Expectation_l,~,~,~]=MarkovChainMoments(z_grid,pi_z);
% If you look at Expectation_l you will see it is now equal to 1
Params.Expectation_l=Expectation_l;

%% Grids

% In the absence of idiosyncratic risk, the steady state equilibrium is given by
r_ss=1/Params.beta-1;
K_ss=((r_ss+Params.delta)/Params.alpha)^(1/(Params.alpha-1)); %The steady state capital in the absence of aggregate uncertainty.

% Set grid for asset holdings
k_grid=10*K_ss*(linspace(0,1,n_k).^3)'; % linspace ^3 puts more points near zero, where the curvature of value and policy functions is higher and where model spends more time

% Bring model into the notational conventions used by the toolkit
d_grid=0; %There is no d variable
a_grid=k_grid;
% pi_z;
% z_grid

n_d=0;
n_a=n_k;
% n_z

% Create functions to be evaluated
FnsToEvaluate.K = @(aprime,a,s) a; %We just want the aggregate assets (which is this periods state)

% Now define the functions for the General Equilibrium conditions
    % Should be written as LHS of general eqm eqn minus RHS, so that the closer the value given by the function is to 
    % zero, the closer the general eqm condition is to holding.
GeneralEqmEqns.CapitalMarket = @(r,K,alpha,delta,Expectation_l) r-(alpha*(K^(alpha-1))*(Expectation_l^(1-alpha))-delta); %The requirement that the interest rate corresponds to the agg capital level
% Inputs can be any parameter, price, or aggregate of the FnsToEvaluate

fprintf('Grid sizes are: %i points for assets, and %i points for exogenous shock \n', n_a,n_z)

%%
DiscountFactorParamNames={'beta'};

ReturnFn=@(aprime, a, s, alpha,delta,mu,r) Aiyagari1994_ReturnFn(aprime, a, s,alpha,delta,mu,r);
% The first inputs must be: next period endogenous state, endogenous state, exogenous state. Followed by any parameters

%%

% Use the toolkit to find the equilibrium price index
GEPriceParamNames={'r'};
% Set initial value for interest rates (Aiyagari proves that with idiosyncratic
% uncertainty, the eqm interest rate is limited above by it's steady state value
% without idiosyncratic uncertainty, that is that r<r_ss).
Params.r=0.038;

% Equilibrium wage
Params.w=(1-Params.alpha)*((Params.r+Params.delta)/Params.alpha)^(Params.alpha/(Params.alpha-1));


%% Run value function
vfoptions.verbose = 1; 

fprintf('Calculating various equilibrium objects \n')
[V,Policy]=ValueFnIter_Case1(n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, [], vfoptions);


%% Run distribution
simoptions.verbose   = 0;
simoptions.maxit     = 50000;
simoptions.tolerance = 1e-6;

% Default method, implemented by Robert
%simoptions.method = 'AD'; % Choose 'RK' or 'AD'

simoptions.method = 'RK';
tic
StationaryDist_RK=StationaryDist_Case1(Policy,n_d,n_a,n_z,pi_z, simoptions);
time_RK = toc;
fprintf('RK: Time to solve for distribution: %f \n',time_RK)


simoptions.method = 'AD';
tic
StationaryDist_AD=StationaryDist_Case1(Policy,n_d,n_a,n_z,pi_z, simoptions);
time_AD = toc;
fprintf('AD: Time to solve for distribution: %f \n',time_AD)
fprintf('Speed-up of AD vs RK: %f \n',time_RK/time_AD)

err_RK_AD = max(abs(StationaryDist_RK(:)-StationaryDist_AD(:)));

fprintf('Discrepancy between two method: %f \n', err_RK_AD)

StationaryDist_RK_a = get_marginal(StationaryDist_RK);
StationaryDist_AD_a = get_marginal(StationaryDist_AD);
Policy         = squeeze(gather(Policy));

PolicyVal = a_grid(Policy);

figure
plot(a_grid,a_grid,'--','LineWidth',2)
hold on
plot(a_grid,PolicyVal(:,1),'LineWidth',2)
hold on
plot(a_grid,PolicyVal(:,end),'LineWidth',2)
legend('45 line','low z','high z')
title('Policy function assets a')

figure
plot(a_grid,StationaryDist_RK_a,'LineWidth',2)
hold on
plot(a_grid,StationaryDist_AD_a,'LineWidth',2)
legend('RK','AD')
title('Stationary Distribution a, PDF')

% %%
% % Solve for the stationary general equilbirium
% vfoptions=struct(); % Use default options for solving the value function (and policy fn)
% simoptions=struct(); % Use default options for solving for stationary distribution
% heteroagentoptions.verbose=1; % verbose means that you want it to give you feedback on what is going on
% 
% fprintf('Calculating price vector corresponding to the stationary general eqm \n')
% [p_eqm,~,GeneralEqmCondn]=HeteroAgentStationaryEqm_Case1(n_d, n_a, n_z, 0, pi_z, d_grid, a_grid, z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Params, DiscountFactorParamNames, [], [], [], GEPriceParamNames,heteroagentoptions, simoptions, vfoptions);
% 
% p_eqm % The equilibrium values of the GE prices
% 
% %% Now that we have the GE, let's calculate a bunch of related objects
% Params.r=p_eqm.r; % Put the equilibrium interest rate into Params so we can use it to calculate things based on equilibrium parameters
% 
% % Equilibrium wage
% Params.w=(1-Params.alpha)*((Params.r+Params.delta)/Params.alpha)^(Params.alpha/(Params.alpha-1));
% 
% fprintf('Calculating various equilibrium objects \n')
% [V,Policy]=ValueFnIter_Case1(n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, [], vfoptions);
% % V is value function
% % Policy is policy function (but as an index of k_grid, not the actual values)
% 
% % PolicyValues=PolicyInd2Val_Case1(Policy,n_d,n_a,n_s,d_grid,a_grid); % This will give you the policy in terms of values rather than index
% 
% StationaryDist=StationaryDist_Case1(Policy,n_d,n_a,n_z,pi_z, simoptions);
% 
% AggVars=EvalFnOnAgentDist_AggVars_Case1(StationaryDist, Policy, FnsToEvaluate,Params, [],n_d, n_a, n_z, d_grid, a_grid,z_grid);
% 
% % AggVars contains the aggregate values of the 'FnsToEvaluate' (in this model aggregates are equal to the mean expectation over the agent distribution)
% % Currently the only FnsToEvaluate is assets, so we get aggregate capital stock
% AggVars.K.Mean
% 
% % Calculate savings rate:
% % We know production is Y=K^{\alpha}L^{1-\alpha}, and that L=1
% % (exogeneous). Thus Y=K^{\alpha}.
% % In equilibrium K is constant, so aggregate savings is just depreciation, which
% % equals delta*K. The agg savings rate is thus delta*K/Y.
% % So agg savings rate is given by s=delta*K/(K^{\alpha})=delta*K^{1-\alpha}
% aggsavingsrate=Params.delta*(AggVars.K.Mean)^(1-Params.alpha);
% 
% % Calculate Lorenz curves, Gini coefficients, and Pareto tail coefficients
% FnsToEvaluateIneq.Earnings = @(aprime,a,z,w) w*z;
% FnsToEvaluateIneq.Income = @(aprime,a,z,r,w) w*z+(1+r)*a;
% FnsToEvaluateIneq.Wealth = @(aprime,a,s) a;
% LorenzCurves=EvalFnOnAgentDist_LorenzCurve_Case1(StationaryDist, Policy, FnsToEvaluateIneq, Params,[], n_d, n_a, n_z, d_grid, a_grid, z_grid);
% 
% % 3.5 The Distributions of Earnings and Wealth
% %  Gini for Earnings
% EarningsGini=Gini_from_LorenzCurve(LorenzCurves.Earnings);
% IncomeGini=Gini_from_LorenzCurve(LorenzCurves.Income);
% WealthGini=Gini_from_LorenzCurve(LorenzCurves.Wealth);
% 
% % Calculate inverted Pareto coeff, b, from the top income shares as b=1/[log(S1%/S0.1%)/log(10)] (formula taken from Excel download of WTID database)
% % No longer used: Calculate Pareto coeff from Gini as alpha=(1+1/G)/2; ( http://en.wikipedia.org/wiki/Pareto_distribution#Lorenz_curve_and_Gini_coefficient)
% % Recalculte Lorenz curves, now with 1000 points
% LorenzCurves=EvalFnOnAgentDist_LorenzCurve_Case1(StationaryDist, Policy, FnsToEvaluateIneq, Params,[], n_d, n_a, n_z, d_grid, a_grid, z_grid, [],1000);
% EarningsParetoCoeff=1/((log(LorenzCurves.Earnings(990))/log(LorenzCurves.Earnings(999)))/log(10)); %(1+1/EarningsGini)/2;
% IncomeParetoCoeff=1/((log(LorenzCurves.Income(990))/log(LorenzCurves.Income(999)))/log(10)); %(1+1/IncomeGini)/2;
% WealthParetoCoeff=1/((log(LorenzCurves.Wealth(990))/log(LorenzCurves.Wealth(999)))/log(10)); %(1+1/WealthGini)/2;
% 
% 
% %% Display some output about the solution
% 
% % plot(cumsum(sum(StationaryDist,2))) %Plot the asset cdf
% 
% fprintf('For parameter values sigma=%.2f, mu=%.2f, rho=%.2f \n', [Params.sigma,Params.mu,Params.rho])
% fprintf('The table 1 elements are sigma=%.4f, rho=%.4f \n',[sqrt(z_variance), z_corr])
% 
% fprintf('The equilibrium value of the interest rate is r=%.4f \n', p_eqm.r*100)
% fprintf('The equilibrium value of the aggregate savings rate is s=%.4f \n', aggsavingsrate)


function distrib_a = get_marginal(distrib)

distrib = gather(distrib);
distrib_a = sum(distrib,2);

end
