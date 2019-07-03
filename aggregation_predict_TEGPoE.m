function [result,MSLL1,t_predict] = aggregation_predict_TEGPoE(Xt,models,y,yt,q)
% Aggregation GP for prediction
% Inputs:
%        Xt: a nt*d matrix containing nt d-dimensional test points
%        models: a cell structure that contains the sub-models built on subsets, where models{i} is the i-th model
%                 fitted to {Xi and Yi} for 1 <= i <= M
%        criterion: an aggregation criterion to combine the predictions from M sub-models
%                   'PoE': product of GP experts
%                   'GPoE': generalized product of GP experts
%                   'BCM': Bayesian committee machine
%                   'RBCM': robust Bayesian committee machine
%                   'GRBCM': generalized robust Bayesian committee machine
%                   'NPAE': nested pointwise aggregation of experts
% Outputs:
%         mu: a nt*1 vector that represents the prediction mean at nt test points
%         s2: a nt*1 vector that represents the prediction variance at nt test points
%         t_predict: computing time for predictions
%
% H.T. Liu 2018/06/01 (htliu@ntu.edu.sg)

nt = size(Xt,1) ;  % number of test points
M = models{1}.Ms ; % number of experts

% normalization of test points Xt
if strcmp(models{1}.optSet.Xnorm,'Y')
    Xt = (Xt - repmat(models{1}.X_mean,nt,1)) ./ (repmat(models{1}.X_std,nt,1)) ;
end

% predictions of each submodel
t1 = clock ;

    for i = 1:M   
        [mu_experts{i},s2_experts{i}] = gp(models{i}.hyp,models{i}.inffunc,models{i}.meanfunc, ...
                                   models{i}.covfunc,models{i}.likfunc,models{i}.X_norm,models{i}.Y_norm,Xt);
    end


% use an aggregation criterion to combine predictions from submodels

    
    
    
    
        kss = feval(models{1}.covfunc,models{1}.hyp.cov,Xt,'diag') + exp(2*models{1}.hyp.lik); % because s2_experts consider noise

        
%         for i = 1:M
%             beta{i} = exp(0.5*(log(kss) - log(s2_experts{i}))) ;
%             beta_total = beta_total + beta{i} ;
%         end
%         for i = 1:M
%             beta{i} = beta{i} ./ beta_total ;
%         end
     result = [];
     MSLL1  = [];
     
        for q = 1.01:0.01:10
         beta_total = zeros(nt,1) ;
         mu = zeros(nt,1) ; s2 = zeros(nt,1) ;
%          s2p = 0; s2y = 0; mup = 0; muy = 0;
%          lossp = zeros(nt,1); loss = zeros(nt,1);
        for i = 1:M
            %beta{i} = 1/M*ones(length(s2_experts{i}),1) ;
            %beta{i} = 0.5*(log(kss) - log(s2_experts{i})) ;
            beta{i} = sqrt(2*pi) / (sqrt(q)*(q-1)) * (sqrt(s2_experts{i}) ./ (sqrt(2*pi)*sqrt(s2_experts{i})).^q - ...
                           sqrt(kss) ./ (sqrt(2*pi)*sqrt(kss)).^q) ;
            s2 = s2 + beta{i}./s2_experts{i} ; 
        end
        s2 = 1./s2 ;

        for i = 1:M 
            mu = mu + s2.*(beta{i}.*mu_experts{i}./s2_experts{i}) ;
        end
        if strcmp(models{1}.optSet.Ynorm,'Y')
            mu = mu*models{1}.Y_std + models{1}.Y_mean ;
            s2 = s2*(models{1}.Y_std)^2 ;
        end
        
        
        result(end+1) = mse(mu, yt);
        s2p   = 0.5*mean(log(2*pi*s2));
        s2y    = 0.5*log(2*pi*std(y)^2);
        lossp = (yt-mu).^2;
        mup   = 0.5*mean(lossp./s2);
        dt    = size(Xt,1);
        muv   = ones(dt,1)*mean(y);
        loss  = (yt-muv).^2;
        muy    = 0.5*mean(loss/(std(y)^2));
        MSLL1(end+1)  = s2p-s2y+mup-muy;
        end   
    
     



t2 = clock ;
t_predict = etime(t2,t1) ;

end