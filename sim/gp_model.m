function [mu, var] = gp_model(omega_train, omega_star, post, hyperparams, covfunc)
%gp_model - Prediction on an input omega
%   Returns the mean and the variance of the output

sn2 = exp(2*hyperparams.lik);  % process noise (optimized likelihood hyperparam: log of the noise standard deviation)

alpha = post.alpha;         % linear parameters encapsulating (K+sn2*I)^(-1)*Y
L = post.L;                 % Cholesky factorization
W = post.sW;                % vector holding the precision (inv cov) of process noise 
Ks = feval(covfunc, hyperparams.cov, omega_train, omega_star);      % cross co-variance
kss = feval(covfunc, hyperparams.cov, omega_star, 'diag');                  % self-variance

V  = L'\(repmat(W,1,length(omega_star)).*Ks);   % contains chol decomp => use Cholesky parameters (alpha,sW,L)
fs2 = kss - sum(V.*V,1)';                       % predicted latent variance
var = fs2 + sn2;                                % predicted output variance
mu = Ks'*alpha;                                 % predicted output mean (equals to predicted latent mean)

end
