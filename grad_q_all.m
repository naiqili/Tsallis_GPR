function q = grad_q_all(models, criterion, opts, iter=50, bs=100, lr=0.00000001, lambda=100000000, init_q=1.5)
    M = length(models); % M experts
    q = init_q;
    mi = 1;
    md = models{mi};
    Xt = md.X([1, 2], :); % randomly select test data
    Yt = md.Y([1, 2], :); % randomly select test data   
    bs = 2;
    nt = 2;
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
    end
    while true
        if iter == 0
            break
        endif
        printf("Optimizing q: remain iter - %d, q - %6.4f\r\n", iter, q);
        md = models{mi};
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
                    %beta{i} = sqrt(2*pi) / (sqrt(q)*(q-1)) * (sqrt(s2_experts{i}) ./ (sqrt(2*pi)*sqrt(s2_experts{i})).^q - sqrt(kss) ./ (sqrt(2*pi)*sqrt(kss)).^q) ;
                endif
                beta_total = beta_total + beta{i} ;

                s2 = s2 + beta{i}./s2_experts{i} ; 
            end
            s2 = 1./(s2 + (1-beta_total)./kss) ;

            for i = 1:M 
                mu = mu + s2.*(beta{i}.*mu_experts{i}./s2_experts{i}) ;
            end
        end
        grad_q_norm = 0.0; grad_q_reg = 0.0; grad_q = 0.0; lambda = 10000000;
        for j = 1:M
            if j != mi
                for k = 1:length(s2)
                    db = log(sqrt(s2(k))) - log(sqrt(s2_experts{j}(k))) + 0.5 * sum((Yt - mu).^2) / s2(k) - 0.5 * sum((Yt - mu_experts{j}(k)).^2) / s2_experts{j}(k) ;
                    ss = sqrt(kss(k));
                    si = sqrt(s2_experts{j}(k));
                    dq = 1 / ((-1+q)^2 * q^1.5) * 2^(-0.5-q) * pi^(0.5-q) * si^(-q) * ss^(-q) * (-(-si^q * ss + si * ss^q)*(2^(1+0.5*q)*pi^(0.5*q)*q + (2*pi)^(0.5*q)*(-1+q)*(1+q*log(2*pi))) - 2^(1+0.5*q)*pi^(0.5*q)*(-1+q)*q*si*ss^q*log(si) + 2^(1+0.5*q)*pi^(0.5*q)*(-1+q)*q*si^q*ss*log(ss));
                    grad_q_norm += db * dq;
                    grad_q_reg -= lambda * beta{j}(k) * dq;
                endfor
            endif
        endfor
        grad_q = grad_q_reg;
        grad_q /= (bs*(M-1));
        % update q
        del_q = lr * grad_q;
        clip = 0.8;
        del_q = min(del_q, clip);
        del_q = max(del_q, -clip);
        q = q + del_q;
        btt = 0.0;
        for j = 1:M
            if j != mi
                for k = 1:length(s2)
                    btt += beta{j}(k)^2;
                endfor
            endif
        endfor
        btt
        iter -= 1;
    endwhile