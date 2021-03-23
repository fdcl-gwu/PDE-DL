function [  ] = fmatch_auto(  )

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
decayRate = 1e-3;

% initialize parameters
parameters = struct;

parameters.fc1.Weights = dlarray(sqrt(2)*randn([Nn(1),1]));
parameters.fc1.Bias = dlarray(zeros(Nn(1),1));

if Nl >= 2
    for nl = 2:Nl
        parameters.("fc"+nl).Weights = dlarray(sqrt(Nn(nl-1))*randn(Nn(nl),Nn(nl-1)));
        parameters.("fc"+nl).Bias = dlarray(zeros(Nn(nl),1));
    end
end

parameters.("fc"+(Nl+1)).Weights = dlarray(sqrt(Nn(Nl))*randn(1,Nn(Nl)));
parameters.("fc"+(Nl+1)).Bias = dlarray(zeros(1));

% random points
x = (rand(Ns,1)-0.5)*Lx;
fTrue = p(x);

ds = arrayDatastore([x,fTrue]);
mbq = minibatchqueue(ds,'MiniBatchSize',1000,'MiniBatchFormat','BC','OutputEnvironment','CPU');

% train
iter = 0;
averageGrad = [];
averageSqGrad = [];

loss = zeros(Ns/1000,1);

while hasdata(mbq)
    iter = iter+1;
    
    data = next(mbq);
    dl_x = data(1,:);
    dl_fTrue = data(2,:);
    
    [dL,loss(iter)] = dlfeval(@get_df,parameters,dl_x,dl_fTrue);
    
    k = initialLearnRate / (1+decayRate*iter);
    [parameters,averageGrad,averageSqGrad] = adamupdate(parameters,dL,...
        averageGrad,averageSqGrad,iter,k);
end

% estimate
fEst = extractdata(get_f(parameters,dlarray(x_grid,'CB')));

% true value
fTrue = p(x_grid);

% plot
figure; hold on;
plot(x_grid,fTrue);
plot(x_grid,fEst);

figure;
plot(loss);

end


function [ f ] = get_f(parameters, x)

Nl = numel(fieldnames(parameters))-1;

weight = parameters.fc1.Weights;
bias = parameters.fc1.Bias;
f = fullyconnect(x,weight,bias);

for nl = 2:Nl+1
    f = sigmoid(f);
    weight = parameters.("fc"+nl).Weights;
    bias = parameters.("fc"+nl).Bias;
    f = fullyconnect(f,weight,bias);
end

end


function [ dL, loss ] = get_df(parameters, x, fTrue)

f = get_f(parameters, x);

loss = mse(f,fTrue);
dL = dlgradient(loss,parameters);

end



