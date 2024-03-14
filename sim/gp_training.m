function [post, hyp, covfunc] = gp_training(x_train, y_train)
%gp_model - Train a GP and returns the mean and variance function

meanfunc = @meanZero;               % empty: don't use a mean function
covfunc = @covSEiso;                % Squared Exponental covariance function
likfunc = @likGauss;                % Gaussian likelihood
inferfunc = @infGaussLik;           % Exact inference

% Optimize hyperparams
hyp = struct('mean', [], 'cov', [0 0], 'lik', -1);
hyp = minimize(hyp, @gp, -100, inferfunc, meanfunc, covfunc, likfunc, x_train, y_train);

% Do inference to retrieve posterior of the latent variables of test inputs
[post nlZ dnlZ] = feval(inferfunc, hyp, {meanfunc}, {covfunc}, {likfunc}, x_train, y_train);

end
