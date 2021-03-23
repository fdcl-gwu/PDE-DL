function [  ] = fmatch(  )

% reference function
miu = -2;
a = 0.5;
sigma = 1;
x0 = 0;
t0 = 0.025;
D = sigma^2/2;

p = @(x)sqrt(a/(2*pi*D*(1-exp(-2*a*t0))))*exp(-a/(2*D).*...
    (x-miu-(x0-miu)*exp(-a*t0)).^2/(1-exp(-2*a*t0)));

% grid
Lx = 10;
Nx = 101;
x_grid = linspace(-Lx/2,Lx/2,Nx);

% number of layers
Nl = 2;

% number of nodes
Nn = [10,10];

% number of samplex
Ns = 1000000;

% learning rate
initialLearnRate = 0.1;
decayRate = 1e-6;

% initialize parameters
W = cell(Nl+1,1);
W{1} = (rand(Nn(1),1)-0.5)*2;
if Nl >= 2
    for nl = 1:Nl-1
        W{nl+1} = (rand(Nn(nl+1),Nn(nl))-0.5)*2/sqrt(Nn(nl));
    end
end
W{end} = (rand(1,Nn(end))-0.5)*2/sqrt(Nn(end));

c = cell(Nl+1,1);
for nl = 1:Nl
    c{nl} = zeros(Nn(nl),1);
end
c{end} = 0;

% allocate memory
G = zeros(Ns,1);

psi = cell(Nl+1,1);
dpsi = cell(Nl,1);
dfdZ = cell(Nl+1,1);
dfdW = cell(Nl+1,1);
dfdc = cell(Nl+1,1);
dGdW = cell(Nl+1,1);
dGdc = cell(Nl+1,1);

% stochastic gradient descent
for iter = 1:Ns
    % generate random samples
    x = (rand-0.5)*Lx;
    
    % reference function
    px = p(x);
    
    % forward propagation
    psi{1} = x;
    for nl = 1:Nl
        Z = W{nl}*psi{nl}+c{nl};
        eZ = exp(Z);
        psi{nl+1} = eZ./(1+eZ);
        dpsi{nl} = psi{nl+1}.*(1-psi{nl+1});
    end
    f = W{end}*psi{end}+c{end};
    
    % backward propagation
    dfdZ{Nl+1} = 1;
    for nl = Nl:-1:1
        dfdZ{nl} = W{nl+1}'*dfdZ{nl+1}.*dpsi{nl};
    end
    
    for nl = 1:Nl+1
        dfdc{nl} = dfdZ{nl};
        dfdW{nl} = dfdZ{nl}*psi{nl}';
        
        dGdc{nl} = 2*(f-px)*dfdc{nl};
        dGdW{nl} = 2*(f-px)*dfdW{nl};
    end
    
    % gradient descent
    k = initialLearnRate / (1+decayRate*iter);
    for nl = 1:Nl+1
        W{nl} = W{nl} - k*dGdW{nl};
        c{nl} = c{nl} - k*dGdc{nl};
    end
    
    G(iter) = (f-px)^2;
end

% estimated values
f = zeros(Nx,1);
for nx = 1:Nx
    f(nx) = get_f(Nl,W,c,x_grid(nx));
end

% plot
figure;
plot(G);

figure; hold on;
plot(x_grid,p(x_grid));
plot(x_grid,f);

end


function [ f ] = get_f( Nl, W, c, x )

ez = exp(W{1}*x + c{1});
psi = cell(Nl,1);
for nl = 1:Nl
    psi{nl} = ez./(1+ez);
    ez = exp(W{nl+1}*psi{nl} + c{nl+1});
end
f = W{Nl+1}*psi{Nl} + c{nl+1};

end

