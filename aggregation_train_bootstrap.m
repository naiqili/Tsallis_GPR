function [models,t_train] = aggregation_train_bootstrap(X,Y,opts)
% Aggeration GP for large scale training data
% 
% Inputs:
%        X: a n*d matrix comprising n d-dimensional training points
%        Y: a n*1 vector containing the function responses of n training points
%        opts: options to build distributed GP models
%             .Xnorm: 'Y' normalize X along each coordiante to have zero mean and unite variance
%             .Ynorm: 'Y' normalize Y to have zero mean and unite variance
%             .Ms: number of experts
%             .partitionCriterion: 'random', 'kmeans'
%             .meanfunc, .covfunc, .likfunc, .inffunc: GP configurations
%             .ell, .sf2, .sn2: parameters for the SE covariance function
%             .numOptFC: optimization setting for min-NLML
% Outputs:
%         models: a cell structure that contains the sub-models built on subsets, where models{i} is the i-th model
%                 fitted to {Xi and Yi} for 1 <= i <= M
%         t_train: computing time for min-NLML optimization
%
% H.T. Liu 2018/06/01 (htliu@ntu.edu.sg)

% Normalize training data
[n,d] = size(X) ;
x_train_mean = zeros(1,d) ; x_train_std  = ones(1,d) ;
y_train_mean = 0 ; y_train_std  = 1 ;
if strcmp(opts.Xnorm,'Y'); x_train_mean = mean(X) ; x_train_std  = std(X) ; end
x_train = (X-repmat(x_train_mean,n,1)) ./ repmat(x_train_std,n,1) ;    
if strcmp(opts.Ynorm,'Y'); y_train_mean = mean(Y) ; y_train_std  = std(Y) ; end
y_train = (Y-y_train_mean)/y_train_std ;

% Partition training data into M subsets
M = opts.Ms;
Msize = opts.Msize;
[x_trains,y_trains,Xs,Ys] = partitionData(x_train,y_train,X,Y,M,Msize,opts.partitionCriterion,opts.covfunc) ;

% Infer hyperparameters by a PoE (product-of-experts) model
meanfunc = opts.meanfunc ; covfunc  = opts.covfunc ; likfunc  = opts.likfunc ; inffunc  = opts.inffunc ;
ell = opts.ell ; sf2 = opts.sf2 ; sn2 = opts.sn2 ;
hyp = struct('mean', [], 'cov', [ones(d,1)*log(ell);log(sqrt(sf2))], 'lik', log(sqrt(sn2)));
numOptFC = opts.numOptFC ;
t1 = clock ; hyp_opt = minimize(hyp, @gp_factorise, numOptFC, inffunc, meanfunc, covfunc, likfunc, x_trains, y_trains); t2 = clock ;
t_train = etime(t2,t1) ;

% Export models
for i = 1:M 
    % different for the M GP experts
    model.X = Xs{i} ; model.Y = Ys{i} ;
    model.X_norm = x_trains{i} ; model.Y_norm = y_trains{i} ;    
    % same for the M GP experts
    model.hyp = hyp_opt ;
    model.X_mean = x_train_mean ; model.X_std = x_train_std ;
    model.Y_mean = y_train_mean ; model.Y_std = y_train_std ;
    model.meanfunc = meanfunc ; model.covfunc = covfunc ; model.likfunc = likfunc ; model.inffunc = inffunc ;
    model.optSet = opts;
    model.Ms = opts.Ms ;
  
    models{i} = model ;
end

end

%(x_train,y_train,X,Y,M,opts.partitionCriterion,opts.covfunc,hyp_knkmeans)
%--------------------------------------------------------
function [xs,ys,Xs,Ys] = partitionData(x,y,X,Y,M,Msize,partitionCriterion,cov)
% Random or disjoint partition of data
% x, y - normalized training data
% X, Y - original training data
% M    - number of subsets
% Msize- size of each expert

[n,d] = size(x) ;
if M > n; warning('The partition number M exceeds the number of training points.'); end

switch partitionCriterion
    case  'random'
        A = unidrnd(n, M, Msize) ;
        for i = 1:M
            xs{i} = x(A(i,:),:) ; ys{i} = y(A(i,:)) ;
            Xs{i} = X(A(i,:),:) ; Ys{i} = Y(A(i,:)) ;
        end
    case 'dirtyKmeans'
        tx = x';
        mu = tx(:,randperm(n,M));    % randomly select M points as centers
        [~, A] = sort(repmat(dot(mu,mu,1)'/2,1,n)-mu'*x', 2);   % A: M x n
        for i = 1:M
            xs{i} = x(A(i,1:Msize),:) ; ys{i} = y(A(i,1:Msize)) ;
            Xs{i} = X(A(i,1:Msize),:) ; Ys{i} = Y(A(i,1:Msize)) ;
        end
    otherwise
        error('No such partition criterion.') ;
end
temp = [];
for i=1:M
    temp(i) = size(xs{i},1);
end
end 


function [nlZ,dnlZ] = gp_factorise(hyp,inf,mean,cov,lik,xs,ys)
% Factorised NLML
% -logp(Y|X,theta) = -\sum_{k=1}^M logp_k(Y^k|X^K,theta)

M = length(xs) ; d = size(xs{1},2) ;
nlZ = zeros(M,1) ;
dnlZ.mean = [] ; cov_grad = zeros(numel(hyp.cov),M) ; lik_grad = zeros(M,1) ;

for i = 1:M 
    x = xs{i} ; y = ys{i} ;
    [nlZ_i,dnlZ_i] = gp(hyp,inf,mean,cov,lik,x,y) ;
    nlZ(i) = nlZ_i ;
    cov_grad(:,i) = dnlZ_i.cov ; lik_grad(i) = dnlZ_i.lik ;
end

nlZ = sum(nlZ) ; dnlZ.cov = sum(cov_grad,2) ; dnlZ.lik = sum(lik_grad) ;
end

