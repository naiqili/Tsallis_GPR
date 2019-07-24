function q = grad_q(models, criterion, optSet, iter=200, bs=100, lr=0.0000000002, lambda=100, init_q=2.5)
    M = length(models); % M experts
    q = init_q;
    while true
        if iter == 0
            break
        endif
        printf("Optimizing q: remain iter - %d, q - %6.4f\r\n", iter, q);
        mi = randi(M);
        md = models{mi};
        rnd_idx = randperm(size(md.X, 1));
        nt = min(bs, size(md.X, 1));
        Xt = md.X(rnd_idx(1:nt), :); % randomly select test data
        Yt = md.Y(rnd_idx(1:nt), :); % randomly select test data   
        % normalization of test points Xt
        if strcmp(models{1}.optSet.Xnorm,'Y')
            Xt = (Xt - repmat(models{1}.X_mean,nt,1)) ./ (repmat(models{1}.X_std,nt,1)) ;
        end
        % normalization of test points Yt
        if strcmp(models{1}.optSet.Ynorm,'Y')
            Yt = (Yt - repmat(models{1}.Y_mean,nt,1)) ./ (repmat(models{1}.Y_std,nt,1)) ;
        end
        for i = 1:M   
            [mu_experts{i},s2_experts{i}] = gp(models{i}.hyp,models{i}.inffunc,models{i}.meanfunc, ...
                                       models{i}.covfunc,models{i}.likfunc,models{i}.X_norm,models{i}.Y_norm,Xt);
           for k = 1:length(s2_experts{i})
               if s2_experts{i}(k) < 0.0000001
                   s2_experts{i}(k) = 0.0000001;
               endif
           endfor
        end
        mu = zeros(nt,1) ; s2 = zeros(nt,1) ;
        switch criterion 
        case 'TERBCM' % robust Bayesian committee machine
            kss = feval(models{1}.covfunc,models{1}.hyp.cov,Xt,'diag') + exp(2*models{1}.hyp.lik); % because s2_experts consider noise
            beta_total = zeros(nt,1) ;
            mu = zeros(nt,1) ; s2 = zeros(nt,1) ;
            for i = 1:M
                if q == 1.0                
                    beta{i} = 0.5*(log(kss) - log(s2_experts{i})) ;
                else
                    beta{i} = 1 / (q-1) * (q^-0.5 * (sqrt(s2_experts{i}) .* sqrt(2*pi)).^(1-q) - q^-0.5 * (sqrt(kss) .* sqrt(2*pi)).^(1-q));
                endif
                beta_total = beta_total + beta{i} ;

                s2 = s2 + beta{i}./s2_experts{i} ; 
            end
            s2 = 1./(s2 + (1-beta_total)./kss) ;

            for i = 1:M 
                mu = mu + s2.*(beta{i}.*mu_experts{i}./s2_experts{i}) ;
            end
        case 'TEGRBCM'
            kss = feval(models{1}.covfunc,models{1}.hyp.cov,Xt,'diag') + exp(2*models{1}.hyp.lik); % because s2_experts consider noise

            for i = 1:M
                if i == 1
                    models_cross{i} = models{i} ;
                else
                    model = models{i} ;
                    model.X = [models{1}.X;models{i}.X] ; model.Y = [models{1}.Y;models{i}.Y] ; % X1 + Xi % Y1 + Yi
                    model.X_norm = [models{1}.X_norm;models{i}.X_norm] ; model.Y_norm = [models{1}.Y_norm;models{i}.Y_norm] ;

                    models_cross{i} = model ;
                end
            end

            for i = 1:M                            
                [mu_crossExperts{i},s2_crossExperts{i}] = gp(models_cross{i}.hyp,models_cross{i}.inffunc,models_cross{i}.meanfunc, ...
                                        models_cross{i}.covfunc,models_cross{i}.likfunc,models_cross{i}.X_norm,models_cross{i}.Y_norm,Xt);
            end

            % combine predictions from GP experts
            beta_total = zeros(nt,1) ;
            %zero_num = zeros(M,1);
            for i = 1:M
                if i > 2
                    if q == 1.0                
                        beta{i} = 0.5*(log(s2_crossExperts{1}) - log(s2_crossExperts{i})) ;
                    else
                        beta{i} = 1 / (q-1) * (q^-0.5 * (sqrt(s2_crossExperts{i}) .* sqrt(2*pi)).^(1-q) - q^-0.5 * (sqrt(s2_crossExperts{1}) .* sqrt(2*pi)).^(1-q));
                    endif
                else 
                    beta{i} = ones(nt,1) ; % beta_1 = beat_2 = 1 ;
                end
                beta_total = beta_total + beta{i} ;
                s2 = s2 + beta{i}./s2_crossExperts{i} ; 
            end

            s2 = 1./(s2 + (1-beta_total)./s2_crossExperts{1}) ;

            for i = 1:M 
                mu = mu + beta{i}.*mu_crossExperts{i}./s2_crossExperts{i} ;
            end
            mu = s2.*(mu + (1-beta_total).*mu_crossExperts{1}./s2_crossExperts{1})  ;
        case 'TEGPoE' % generalized product of GP experts using beta_i = 1/M  Tsallis Entropy
            kss = feval(models{1}.covfunc,models{1}.hyp.cov,Xt,'diag') + exp(2*models{1}.hyp.lik);
            for i = 1:M
                if q == 1.0                
                    beta{i} = 0.5*(log(kss) - log(s2_experts{i})) ;
                else
                    beta{i} = sqrt(2*pi) / (sqrt(q)*(q-1)) * (sqrt(s2_experts{i}) ./ (sqrt(2*pi)*sqrt(s2_experts{i})).^q - ...
                                   sqrt(kss) ./ (sqrt(2*pi)*sqrt(kss)).^q) ;
                endif
                s2 = s2 + beta{i}./s2_experts{i} ; 
            end
            s2 = 1./s2 ;

            for i = 1:M 
                mu = mu + s2.*(beta{i}.*mu_experts{i}./s2_experts{i}) ;
            end
        otherwise
            error('No such aggregation model.') ;
        end
        grad_q_norm = 0.0; grad_q_reg = 0.0; grad_q = 0.0; 
        for j = 1:M
            if strcmp(criterion, 'TEGRBCM') && j==1
                continue
            endif
            if j != mi
                for k = 1:size(nt, 1)
                    ss = sqrt(kss(k));
                    if strcmp(criterion, 'TEGRBCM')
                        si = sqrt(s2_crossExperts{j}(k));
                    else
                        si = sqrt(s2_experts{j}(k));
                    endif
                    if strcmp(criterion, 'TEGPoE')
                        db = - log(si) - 0.5 * sum((Yt - mu_experts{j}).^2) / s2_experts{j}(k);
                    else
                        db = log(ss) - log(si) + 0.5 * sum((Yt - 0).^2) / kss(k) - 0.5 * sum((Yt - mu_experts{j}).^2) / s2_experts{j}(k);
                    endif
                    dq = 1 / ((-1+q)^2 * q^1.5) * 2^(-0.5-q) * pi^(0.5-q) * si^(-q) * ss^(-q) * (-(-si^q * ss + si * ss^q)*(2^(1+0.5*q)*pi^(0.5*q)*q + (2*pi)^(0.5*q)*(-1+q)*(1+q*log(2*pi))) - 2^(1+0.5*q)*pi^(0.5*q)*(-1+q)*q*si*ss^q*log(si) + 2^(1+0.5*q)*pi^(0.5*q)*(-1+q)*q*si^q*ss*log(ss));
                    grad_q_norm += db * dq;
                    grad_q_reg -= beta{j}(k) * dq;
                endfor
            endif
        endfor
        grad_q_norm
        grad_q_reg
        grad_q = grad_q_norm + lambda * grad_q_reg;
        grad_q /= (nt*(M-1));
        % update q
        del_q = lr * grad_q;
        clip = 0.8;
        del_q = min(del_q, clip);
        del_q = max(del_q, -clip);
        q = q + del_q;
        iter -= 1;
    endwhile