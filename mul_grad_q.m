function qs = mul_grad_q(models, criterion, optSet, iter=200, lr=0.0000000002, lambda=100, init_q)
    M = length(models); % M experts
    Msize = zeros(M, 1);
    NN = 0;
    for i=1:M
        Msize(i) = size(models{i}.X, 1);
        NN += Msize(i);
    endfor
    Mprob = Msize / NN;
    qs = init_q;
    last_qs = qs;
    printf("Start preprocessing...\r\n");
    maxM = 0;
    for k=1:M
        maxM = max(maxM, size(models{k}.X, 1));
    endfor
    amu = zeros(M, M, maxM) ; as2 = zeros(M, M, maxM) ; akss = zeros(M, maxM) ; 
    
    if strcmp(criterion,'TEGRBCM') 
        for i=1:M
            if i == 1
                models_cross{i} = models{i} ;
            else
                model = models{i} ;
                model.X = [models{1}.X;models{i}.X] ; model.Y = [models{1}.Y;models{i}.Y] ; % X1 + Xi % Y1 + Yi
                model.X_norm = [models{1}.X_norm;models{i}.X_norm] ; model.Y_norm = [models{1}.Y_norm;models{i}.Y_norm] ;

                models_cross{i} = model ;
            endif
        endfor
    endif
    for k=1:M        
        k
        if ~strcmp(criterion,'TEGRBCM') 
            md = models{k};
            Xt = md.X;
            Yt = md.Y;
            nt = size(Xt, 1);
            % normalization of test points Xt
            if strcmp(models{1}.optSet.Xnorm,'Y')
                Xt = (Xt - repmat(models{1}.X_mean,nt,1)) ./ (repmat(models{1}.X_std,nt,1)) ;
            end
            % normalization of test points Yt
            if strcmp(models{1}.optSet.Ynorm,'Y')
                Yt = (Yt - repmat(models{1}.Y_mean,nt,1)) ./ (repmat(models{1}.Y_std,nt,1)) ;
            end
            models{k}.Xt = Xt; models{k}.Yt = Yt;
            kss = feval(models{1}.covfunc,models{1}.hyp.cov,Xt,'diag') + exp(2*models{1}.hyp.lik);
            kss = max(kss, 1e-4);
            akss(k, 1:length(kss)) = kss;
        else
            md = models_cross{k};
            Xt = md.X;
            Yt = md.Y;
            nt = size(Xt, 1);
            % normalization of test points Xt
            if strcmp(models{1}.optSet.Xnorm,'Y')
                Xt = (Xt - repmat(models{1}.X_mean,nt,1)) ./ (repmat(models{1}.X_std,nt,1)) ;
            end
            % normalization of test points Yt
            if strcmp(models{1}.optSet.Ynorm,'Y')
                Yt = (Yt - repmat(models{1}.Y_mean,nt,1)) ./ (repmat(models{1}.Y_std,nt,1)) ;
            end
            models_cross{k}.Xt = Xt; models_cross{k}.Yt = Yt;
        endif
        for i = 1:M   
            if i == k
                continue
            endif
            if ~strcmp(criterion,'TEGRBCM') 
                [mu_experts{i},s2_experts{i}] = gp(models{i}.hyp,models{i}.inffunc,models{i}.meanfunc, ...
                                           models{i}.covfunc,models{i}.likfunc,models{i}.X_norm,models{i}.Y_norm,Xt);
               s2_experts{i} = max(s2_experts{i}, 1e-4);
               amu(k, i, 1:length(mu_experts{i})) = mu_experts{i}; 
               as2(k, i, 1:length(s2_experts{i})) = s2_experts{i}; 
            else
                 
                [mu_crossExperts{i},s2_crossExperts{i}] = gp(models_cross{i}.hyp,models_cross{i}.inffunc,models_cross{i}.meanfunc, ...
                                        models_cross{i}.covfunc,models_cross{i}.likfunc,models_cross{i}.X_norm,models_cross{i}.Y_norm,Xt);
               s2_crossExperts{i} = max(s2_crossExperts{i}, 1e-4);
               amu(k, i, 1:length(mu_crossExperts{i})) = mu_crossExperts{i}; 
               as2(k, i, 1:length(s2_crossExperts{i})) = s2_crossExperts{i}; 
            endif
        end
    endfor
    printf("Finish preprocessing...\r\n");
    while true
        if iter == 0
            break
        endif
        %qs
        mi = randsample(M, 1, replacement=true, Mprob)(1); 
        if strcmp(criterion, 'TEGRBCM') && (mi == 1 || mi == 2)
            continue
        endif
        if ~strcmp(criterion,'TEGRBCM') 
            mn = size(models{mi}.X, 1);
            b = randi(mn);
            xt = models{mi}.Xt(b); yt = models{mi}.Yt(b); 
        else
            mn = size(models_cross{mi}.X, 1);
            b = randi(mn);
            xt = models_cross{mi}.Xt(b); yt = models_cross{mi}.Yt(b); 
        endif
        grad_q_norm = zeros(M,1); grad_q_reg = zeros(M,1); grad_q = zeros(M,1);  
        for k=1:M
            if k==mi
                continue
            endif
            q=qs(k);
            if strcmp(criterion, 'TEGRBCM')
                if k==1 || k==2
                    continue
                endif
                si = sqrt(as2(mi, 1, b));
                sk = sqrt(as2(mi, k, b)); 
                db = -( log(si) - log(sk) + 0.5 * (yt - amu(mi, 1, b)).^2 / si^2 - 0.5 * (yt - amu(mi, k, b)).^2 / sk^2);
            elseif strcmp(criterion, 'TERBCM')
                si = sqrt(akss(mi, b));
                sk = sqrt(as2(mi, k, b)) ;
                db = -( log(si) - log(sk) + 0.5 * (yt - 0).^2 / si^2 - 0.5 * (yt - amu(mi, k, b)).^2 / sk^2);
            elseif strcmp(criterion, 'TEGPoE')
                si = sqrt(akss(mi, b));
                sk = sqrt(as2(mi, k, b));
                db = -( - log(sk) - 0.5 * (yt - amu(mi, k, b)).^2 / sk^2);
            else
                error('No such criterion.') ;
            endif            
            if q == 1.0                
                beta = 0.5*(log(si^2) - log(sk^2)) ;
            else
                beta = 1 / (q-1) * (q^-0.5 * (sk .* sqrt(2*pi)).^(1-q) - q^-0.5 * (si .* sqrt(2*pi)).^(1-q));
            endif
            dq = 1 / ((-1+q)^2 * q^1.5) * 2^(-0.5-q) * pi^(0.5-q) * sk^(-q) * si^(-q) * (-(-sk^q * si + sk * si^q)*(2^(1+0.5*q)*pi^(0.5*q)*q + (2*pi)^(0.5*q)*(-1+q)*(1+q*log(2*pi))) - 2^(1+0.5*q)*pi^(0.5*q)*(-1+q)*q*sk*si^q*log(sk) + 2^(1+0.5*q)*pi^(0.5*q)*(-1+q)*q*sk^q*si*log(si));
            grad_q_norm(k) += db * dq;
            grad_q_reg(k) += beta * dq;
        endfor
        grad_q = grad_q_norm + lambda * grad_q_reg;
        grad_q /= (nt);
        % update q
        del_q = lr * grad_q;
        clip = 0.8;
        del_q = min(del_q, clip);
        del_q = max(del_q, -clip);
        qs = qs - del_q;
        qs = max(qs, 0.00001);
        %qs_dist = sqrt(sum((qs-last_qs).^2))
        last_qs = qs;
        %ttb = sum(beta_total.^2)
        iter -= 1;
    endwhile