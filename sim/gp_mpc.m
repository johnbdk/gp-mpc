%% Initialize workspace
clear variables;
close all;
clc;

%% Do GP training and predictions on test inputs
num_train = 40;
num_test = 300;

x_test = linspace(-3, 3, num_test)';            % 61 test state training inputs
u_test = linspace(-1, 1, num_test)';            % 61 test input training inputs
x_train = gpml_randn(0.8, num_train, 1);        % 20 training inputs
u_train = linspace(-1.5, 1.1, num_train)';
noise = 0.1*gpml_randn(0.9, num_train, 1);      % 20 noise samples

% Augment the function arguments
omega_train = [x_train, u_train];
omega_test = [x_test, u_test];

[f_train, df_train, f_expr, df_expr] = nominal_model(omega_train);
[h_train, dh_train, h_expr, dh_expr] = true_model(omega_train, noise);
z_train = h_train - f_train;

[post, hyp, covfunc] = gp_training(omega_train, z_train);
[z_mu, z_var, ~, ~, ~] = gp_model(omega_train, omega_test, post, hyp, covfunc);

[f_test] = nominal_model(omega_test);
h_test = f_test + z_mu;

%% Create plot of augmented GP model and the GP model on predicting the deterministic inputs
fig1 = figure('Position', [0 400 1500 1000]);
t = tiledlayout(1,2);
t.Padding = 'compact';
t.TileSpacing = 'compact';

ax1 = nexttile;

conf_bound = [z_mu + 2*sqrt(z_var); flip(z_mu - 2*sqrt(z_var))];
fill3([u_test; flip(u_test)], [x_test; flip(x_test)], conf_bound, 'yellow');
hold on;
plot3(u_test, x_test, z_mu, 'Color', 'blue', 'LineStyle','-', 'LineWidth', 1.5);
z = sin(3*x_test) + u_test;
plot3(u_test, x_test, z, 'Color', 'black', 'LineStyle','--', 'LineWidth', 1.5);
plot3(u_train, x_train, z_train, '+', 'Color', 'red', 'LineWidth', 1.5)
hold off;

xlabel('u'); ylabel('x'); zlabel('h');
title('GP prediction $\hat{z}\bigl(x(k), u(k)\bigr) = g\bigl(x(k),u(k)\bigr) + w(k)$', 'Interpreter', 'latex', 'FontSize', 15);
xlabel('$x(k)$', 'Interpreter', 'latex', 'FontSize', 15);
ylabel('$u(k)$', 'Interpreter', 'latex', 'FontSize', 15);
zlabel('$\hat{z}\bigl(x(k), u(k)\bigr)$', 'Interpreter', 'latex', 'FontSize', 15);
legend('$95\%$ conf. bound','augmented func est', 'true func', '$z(k)$ train samples', ...
   'Location','southeast', 'Interpreter', 'latex', 'FontSize', 15);
view(ax1,[45 0]);

ax2 = nexttile;

conf_bound = [h_test + 2*sqrt(z_var); flip(h_test - 2*sqrt(z_var))];
fill3([u_test; flip(u_test)], [x_test; flip(x_test)], conf_bound, 'yellow');
hold on;
plot3(u_test, x_test, h_test, 'Color', 'blue', 'LineStyle','-', 'LineWidth', 1.5);
[h_test_true] = true_model(omega_test, zeros(length(x_test), 1));
plot3(u_test, x_test, h_test_true, 'Color', 'black', 'LineStyle','--', 'LineWidth', 1.5);
plot3(u_train, x_train, h_train, '+', 'Color', 'red', 'LineWidth', 1.5)
hold off;

xlabel('u'); ylabel('x'); zlabel('h');
title('$x(k+1) = f\bigl(x(k),u(k)\bigr) + \hat{z}\bigl(x(k), u(k)\bigr)$', 'Interpreter', 'latex', 'FontSize', 15);
xlabel('$x(k)$', 'Interpreter', 'latex', 'FontSize', 15);
ylabel('$u(k)$', 'Interpreter', 'latex', 'FontSize', 15);
zlabel('$x(k+1)$', 'Interpreter', 'latex', 'FontSize', 15);
legend('$95\%$ conf. bound','augmented func est', 'true func', '$x(k+1)$ train samples', ...
   'Location','southeast', 'Interpreter', 'latex', 'FontSize', 15);
view(ax2,[45 0]);

%% Mean and Covariance prediction models -- Uncertainty propagation

omega_star = {}; 
omega_star{1} = omega_test(1, :);   % uncertain input mean
omega_star{2} = zeros(2,2);         % uncertain input variance
[mean, cov] = taylor_approx(omega_train, omega_star, post, hyp, covfunc);
