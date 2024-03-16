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

[f_train, df_train, f_expr, df_expr] = nominal_model(x_train, u_train);
[h_train, dh_train, h_expr, dh_expr] = true_model(x_train, u_train, noise);
% 
% figure;
% scatter3(u_train, x_train, h_train);
% xlabel('u'); ylabel('x'); zlabel('h');

z_train = h_train - f_train;
omega_train = [x_train, u_train];
omega_test = [x_test, u_test];

[post, hyp, covfunc] = gp_training(omega_train, z_train);
[z_mu, z_var, alpha, jacobian_tools] = gp_model(omega_train, omega_test, post, hyp, covfunc);

[f_test] = nominal_model(x_test, u_test);
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
[h_test_true] = true_model(x_test, u_test, zeros(length(x_test), 1));
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


%% Uncertainty propagation
omega_star = omega_test(1, :);

[Ks, dKs_] = feval(covfunc, hyp.cov, omega_train, omega_star);
[dhyp, dxs1] = dKs_(1);

tp_expr = jacobian_tools{1};
Ks_expr = jacobian_tools{2};
dKs_expr = jacobian_tools{3};
dKs = double(vpa(subs(subs(dKs_expr, tp_expr, omega_star), Ks_expr, Ks)));

jacobian_mean = dKs'*alpha;
