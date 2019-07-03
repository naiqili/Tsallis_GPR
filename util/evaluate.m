function [SMSE,MSLL,NLPD] = evaluate(mu_dGP,s2_dGP,x,xt,y,yt)
SMSE  = mse(mu_dGP, yt);
s2p   = 0.5*mean(log(2*pi*s2_dGP));
s2    = 0.5*log(2*pi*std(y)^2);
lossp = (yt-mu_dGP).^2;
mup   = 0.5*mean(lossp./s2_dGP);
dt    = size(xt,1);
muv   = ones(dt,1)*mean(y);
loss  = (yt-muv).^2;
mu    = 0.5*mean(loss/(std(y)^2)); 
MSLL  = s2p-s2+mup-mu;

NLPD1 = (mu_dGP - yt).^2 ./ (2 * s2_dGP) ;
NLPD2 = log(sqrt(s2_dGP)) ;
NLPD  = mean(NLPD1 + NLPD2) + 0.5 * log(2 * pi) ; 
end