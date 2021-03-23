function [  ] = ouDL_auto(  )

% parameters
miu = -2;
a = 0.5;
sigma = 1;
x0 = 0;

D = sigma^2/2;

% true density function
p = @(t,x)sqrt(a/(2*pi*D*(1-exp(-2*a*t))))*exp(-a/(2*D).*...
    (x-miu-(x0-miu)*exp(-a*t)).^2/(1-exp(-2*a*t)));

% time
T = 1;
sf = 40;
Nt = T*sf;
t0 = T/sf;
t_grid = linspace(T/sf,1,sf);

% grid
Lx = 10;
Nx = 101;
boundary = [-Lx/2;Lx/2];
x_grid = linspace(-Lx/2,Lx/2,Nx);

% number of layers
Nl = 2;

% number of nodes
Nn = [20,20,20,20,20,20,20,20,20];

% number of batches
Nb = 1000;

% number of samplex
Ns = 50000000;

% learning rate
initialLearnRate = 0.1;
decayRate = 1e-3;

% initialize parameters
parameters = struct;

parameters.fc1.Weights = dlarray(sqrt(2)*randn([Nn(1),2]));
parameters.fc1.Bias = dlarray(zeros(Nn(1),1));

if Nl >= 2
    for nl = 2:Nl
        parameters.("fc"+nl).Weights = dlarray(sqrt(Nn(nl-1))*randn(Nn(nl),Nn(nl-1)));
        parameters.("fc"+nl).Bias = dlarray(zeros(Nn(nl),1));
    end
end

parameters.("fc"+(Nl+1)).Weights = dlarray(sqrt(Nn(Nl))*randn(1,Nn(Nl)));
parameters.("fc"+(Nl+1)).Bias = dlarray(zeros(1));

% random samples
x = (rand(Ns,1)-0.5)*Lx;
t = rand(Ns,1)*T*(sf-1)/sf+T/sf;
y = boundary(randi(2,Ns,1));
tau = rand(Ns,1)*T*(sf-1)/sf+T/sf;
w = (rand(Ns,1)-0.5)*Lx;

f0w_true = p(t0,w);

ds = arrayDatastore([x,t,y,tau,w,f0w_true]);
mbq = minibatchqueue(ds,'MiniBatchSize',Nb,'MiniBatchFormat','BC','OutputEnvironment','CPU');

% training
averageGrad = [];
averageSqGrad = [];
iter = 0;

loss = zeros(3,Ns/Nb);

while hasdata(mbq)
    iter = iter + 1;
    
    data = next(mbq);
    dl_x = data(1,:);
    dl_t = data(2,:);
    dl_y = data(3,:);
    dl_tau = data(4,:);
    dl_w = data(5,:);
    dl_f0w_true = data(6,:);
    
    [dG,loss(1,iter),loss(2,iter),loss(3,iter)] = dlfeval(@get_df,parameters,...
        dl_x,dl_t,dl_y,dl_tau,t0,dl_w,dl_f0w_true,a,miu,sigma);
    
    k = initialLearnRate / (1+decayRate*iter);
    [parameters,averageGrad,averageSqGrad] = adamupdate(parameters,dG,...
        averageGrad,averageSqGrad,iter,k);
end

% estimate
fEst = zeros(Nx,Nt);
for nt = 1:Nt
    fEst(:,nt) = get_f(parameters,dlarray(x_grid,'CB'),...
        dlarray(repmat(t_grid(nt),1,Nx),'CB'));
end

% true function values
fTrue = zeros(Nx,Nt);
for nt = 1:Nt
    fTrue(:,nt) = p(t_grid(nt),x_grid);
end

% plot
figure; hold on;
for nt = 1:Nt
    plot3(x_grid,ones(Nx,1)*t_grid(nt),fTrue(:,nt),'b');
    plot3(x_grid,ones(Nx,1)*t_grid(nt),fEst(:,nt),'r');
end

figure;
plot(loss');

end


function [ f ] = get_f( parameters, x, t )

Nl = numel(fieldnames(parameters))-1;

weight = parameters.fc1.Weights;
bias = parameters.fc1.Bias;
f = fullyconnect([x;t],weight,bias);

for nl = 2:Nl+1
    f = sigmoid(f);
    weight = parameters.("fc"+nl).Weights;
    bias = parameters.("fc"+nl).Bias;
    f = fullyconnect(f,weight,bias);
end

end


function [ dG, G1, G2, G3 ] = get_df(parameters, x, t, y, tau, t0, w, f0w_true, a, miu, sigma)

fxt = get_f(parameters,x,t);

grad_f = dlgradient(sum(fxt,'all'),{x,t},'EnableHigherDerivatives',true);
dfx = grad_f{1};
dft = grad_f{2};

ddfx = dlgradient(sum(dfx,'all'),x,'EnableHigherDerivatives',true);

% differential equation
Lf = dft + a*(-fxt+(miu-x).*dfx) - sigma^2/2*ddfx;
G1 = mse(Lf,zeros(size(Lf),'like',Lf));

% boundary condition
fytau = get_f(parameters,y,tau);
G2 = mse(fytau,zeros(size(fytau),'like',fytau));

% initial condition
f0w = get_f(parameters,w,dlarray(repmat(t0,1,length(w)),'CB'));
G3 = mse(f0w,f0w_true);

loss = G1 + G2 + G3;
dG = dlgradient(loss,parameters);

end

