%% Initialize workspace
clear all;
close all;
clc;

%% Define prediction model and training data
x = gpml_randn(0.8, 20, 1);                                         % 20 training inputs
y = 0.1*x + tanh(x) + sin(3*x) + 0.1*gpml_randn(0.9, 20, 1);        % 20 noisy training targets
z = y - (0.1*x + tanh(x));

xs = linspace(-3, 3, 61)';                          % 61 test inputs
%figure;
%scatter(x,z)
%% Using the build-in framework
meanfunc = @meanZero;               % empty: don't use a mean function
covfunc = @covSEiso;                % Squared Exponental covariance function
likfunc = @likGauss;                % Gaussian likelihood
inferfunc = @infGaussLik;           % Exact inference

% Optimize hyperparams
hyp = struct('mean', [], 'cov', [0 0], 'lik', -1);
hyp2 = minimize(hyp, @gp, -100, inferfunc, meanfunc, covfunc, likfunc, x, z);

%%
sn2 = exp(2*hyp2.lik);  % process noise (optimized likelihood hyperparam: log of the noise standard deviation)
% Do inference to retrieve posterior of the latent variables of test inputs
[post nlZ dnlZ] = feval(inferfunc, hyp2, {meanfunc}, {covfunc}, {likfunc}, x, z);
alpha = post.alpha;     % Linear parameters encapsulating (K+sn2*I)^(-1)*Y
L = post.L;             % Cholesky factorization
W = post.sW;            % vector holding the precision (inv cov) of process noise 
Ks = feval(covfunc, hyp2.cov, x, xs, 'diag');   % cross co-variance
kss = feval(covfunc, hyp2.cov, xs, 'diag');     % self-variance

V  = L'\(repmat(W,1,length(xs)).*Ks);   % contains chol decomp => use Cholesky parameters (alpha,sW,L)
fs2 = kss - sum(V.*V,1)';  % predicted latent variance
ys2 = fs2 + sn2;            % predicted output variance

mu = Ks'*alpha;             % predicted output mean (equals to predicted latent mean)


y_nominal = 0.1*xs + tanh(xs);
y_test = y_nominal + mu;

y_true = y_nominal + sin(3*xs);


f = [y_test + 2*sqrt(ys2); flip(y_test - 2*sqrt(ys2), 1)];
fill([xs; flip(xs,1)], f, 'yellow')
hold on;
plot(xs, y_test, 'Color', 'blue', 'LineStyle','-', 'LineWidth', 1.5);
plot(xs, y_true, 'Color', 'black', 'LineStyle','--', 'LineWidth', 1.5);
plot(x, y, '+', 'Color', 'red', 'LineWidth', 1.5)
hold off;

title('$x(k+1) = f\bigl(x(k),u(k)\bigr) + g\bigl(x(k),u(k)\bigr) + w(k)$', 'Interpreter', 'latex', 'FontSize', 15);
xlabel('$x(k)$', 'Interpreter', 'latex', 'FontSize', 15);
ylabel('$x(k+1)$', 'Interpreter', 'latex', 'FontSize', 15);
legend('$95\%$ conf. bound','augmented func est', 'true func', '$x(k+1)$ train samples', ...
    'Location','southeast', 'Interpreter', 'latex', 'FontSize', 15);
