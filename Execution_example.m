clear
clc

rng(1); % Set a seed for random number generation

N_MC = 10000; % number of Monte Carlo paths to generate
gamma = 5; % range of risk aversion parameter to use (gamma and phi are essentially the same thing)
GG = length(gamma);

% declare initial values of processes and parameters
S_0 = 100;
X_0 = 0;
Q_0 = 10;

b = 0.01;
k = 0.1;
sigma = 1;
rho = 0;

alpha = 0.5;

T = 1;
Ndt = 1000;
dt = T/Ndt;
t = 0:dt:T;

% generate Brownian increments (the same increments used for all values of gamma)
dB = sqrt(dt)*randn(N_MC,Ndt);

% results will be stored in a cell structure
results = cell(1,GG);

tic
for(g = 1:GG)
    X = NaN(N_MC,Ndt+1);
    S = NaN(N_MC,Ndt+1);
    
    X(:,1) = X_0;
    S(:,1) = S_0;
    
    % inventory only needs to be one row because it is deterministic, it
    % will not depend on any random elements
    Q = NaN(1,Ndt+1);
    Q(1) = Q_0;
    nu = NaN(1,Ndt);
    
    % note: CAPITAL letters are used to distinguish between model
    % parameters and other constants defined for convenience. Assignment 6
    % has beta as a mean reversion coefficient, but the lecture notes uses
    % BETA for a constant in defining the optimal trading strategy
    BETA = sqrt(2*k*gamma(g)*sigma^2);
    OMEGA = sqrt(2*gamma(g)*sigma^2/k);
    PHI_p = BETA + 2*alpha - b;
    PHI_m = BETA - 2*alpha + b;
    % the function F is equal to (2*f+b)/(2*k), where f is the function
    % which appears in the dynamics value function
    F = BETA/(2*k) * (PHI_m * exp(-OMEGA/2*(T-t)) - PHI_p * exp(OMEGA/2*(T-t)))./(PHI_m * exp(-OMEGA/2*(T-t)) + PHI_p * exp(OMEGA/2*(T-t)));
    
    % construct inventory path (deterministic)
    for(n = 1:Ndt)
        nu(n) = F(n) * Q(n);
        Q(n+1) = Q(n) + nu(n)*dt;
    end
    
    % construct price and cash process paths, both depending on trading
    % speed
    for(n = 2:Ndt+1)
        S(:,n) = S(:,n-1) + b*nu(n-1)*dt + sigma.*dB(:,n-1);
        X(:,n) = X(:,n-1) - (S(:,n-1) + k*nu(n-1)).*nu(n-1)*dt;
    end
    
    % if only one MC path, store the whole result. Otherwise, only store
    % end values (to save on memory).
    if(N_MC == 1)
        results{g}.X = X(1,:);
        results{g}.S = S(1,:);
    else
        results{g}.X = X(:,end);
        results{g}.S = S(:,end);
    end
    
    results{g}.Q = Q(:);
    results{g}.nu = nu(:);
    
    % the value of W at time T is the total wealth of the agent at time T,
    % consisting of cash and penalized liquidation value of remaining
    % inventory
    results{g}.W = X(:,end) + Q(end)*(S(:,end) - alpha*Q(end));
    
end
toc

EW = NaN(1,GG);
SW = NaN(1,GG);

% for each value of gamma compute the expected terminal wealth and standard
% deviation of terminal wealth
for(g = 1:GG)
    EW(g) = mean(results{g}.W);
    SW(g) = std(results{g}.W);
end

%%

% if there is only 1 MC path we want to look at the behavior of the path
if(N_MC == 1)
    fig = figure(1);
    clf(fig)
    axes('fontsize',15,'Position',[0.1589 0.1381 0.7447 0.7869],'Parent',fig);
    hold on
    for(g = 1:GG)
        plot(t, results{g}.Q(:),'linewidth',2,'color',[(g-1)/(GG-1) 0 1-(g-1)/(GG-1)])
    end
    xlabel('$t$','fontsize',15,'interpreter','latex')
    ylabel('$Q_t$','fontsize',15,'interpreter','latex')

    fig = figure(2);
    clf(fig)
    axes('fontsize',15,'Position',[0.1589 0.1381 0.7447 0.7869],'Parent',fig);
    hold on
    for(g = 1:GG)
        plot(t(1:end-1), results{g}.nu(:),'linewidth',2,'color',[(g-1)/(GG-1) 0 1-(g-1)/(GG-1)])
    end
    xlabel('$t$','fontsize',15,'interpreter','latex')
    ylabel('$\nu_t$','fontsize',15,'interpreter','latex')

    fig = figure(3);
    clf(fig)
    axes('fontsize',15,'Position',[0.1589 0.1381 0.7447 0.7869],'Parent',fig);
    hold on
    for(g = 1:GG)
        plot(t, results{g}.S(:),'linewidth',2,'color',[(g-1)/(GG-1) 0 1-(g-1)/(GG-1)])
    end
    xlabel('$t$','fontsize',15,'interpreter','latex')
    ylabel('$S_t$','fontsize',15,'interpreter','latex')
%     ylim([99.5 100.5])

    fig = figure(4);
    clf(fig)
    axes('fontsize',15,'Position',[0.1589 0.1381 0.7447 0.7869],'Parent',fig);
    hold on
    for(g = 1:GG)
        plot(t, results{g}.X(:),'linewidth',2,'color',[(g-1)/(GG-1) 0 1-(g-1)/(GG-1)])
    end
    xlabel('$t$','fontsize',15,'interpreter','latex')
    ylabel('$X_t$','fontsize',15,'interpreter','latex')
else
    % if there are multiple MC paths we want to consider the distribution
    % of terminal wealth
    fig = figure(5);
    t1 = 0:dt:(T-dt);
    clf(fig)
    axes('fontsize',15,'Position',[0.1589 0.1381 0.7447 0.7869],'Parent',fig);
    hold on
    plot(SW, EW,'o-','linewidth',2)
    %plot(t1, nu)
    ylabel('nu value','fontsize',15,'interpreter','latex')
    xlabel('time','fontsize',15,'interpreter','latex')
end