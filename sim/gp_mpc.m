%% Initialize workspace
clear variables;
close all;
clc;

%% Do GP training and predictions on test inputs
num_train = 300;
num_test = 60;

x_test = linspace(-3, 3, num_test)';            % 61 test state training inputs
u_test = linspace(-1, 1, num_test)';            % 61 test input training inputs
x_train = gpml_randn(0.8, num_train, 1);        % 20 training inputs
u_train = linspace(-1.5, 1.1, num_train)';
noise = 0.1*gpml_randn(0.9, num_train, 1);      % 20 noise samples

[f_train, df_train, f_expr, df_expr] = nominal_model(x_train, u_train);
[h_train, dh_train, h_expr, dh_expr] = true_model(x_train, u_train, noise);

figure;
scatter3(u_train, x_train, h_train);
xlabel('u'); ylabel('x'); zlabel('h');

z_train = h_train - f_train;
omega_train = [x_train, u_train];
omega_test = [x_test, u_test];

[post, hyp, covfunc] = gp_training(omega_train, z_train);
[z_mu, z_var] = gp_model(omega_train, omega_test, post, hyp, covfunc);

[f_test] = nominal_model(x_test, u_test);
h_test = f_test + z_mu;

%% Create a plot on predicting the deterministic inputs
[h_test_true] = true_model(x_test, u_test, zeros(length(x_test), 1));

conf_bound = [z_mu + 2*sqrt(z_var); flip(z_mu - 2*sqrt(z_var))];

figure;
fill([x_test; flip(x_test,1)], conf_bound, 'yellow')
fill3([u_test; flip(u_test)], [x_test; flip(x_test)], conf_bound, 'yellow');
hold on;

true = sin(3*x_test) + u_test;
plot3(u_test, x_test, z_mu, 'Color', 'blue', 'LineStyle','-', 'LineWidth', 1.5);
% hold on;
plot3(u_test, x_test, true, 'Color', 'black', 'LineStyle','--', 'LineWidth', 1.5);
xlabel('u'); ylabel('x'); zlabel('h');
plot3(u_train, x_train, z_train, '+', 'Color', 'red', 'LineWidth', 1.5)
hold off;

title('$x(k+1) = f\bigl(x(k),u(k)\bigr) + g\bigl(x(k),u(k)\bigr) + w(k)$', 'Interpreter', 'latex', 'FontSize', 15);
xlabel('$x(k)$', 'Interpreter', 'latex', 'FontSize', 15);
ylabel('$u(k)$', 'Interpreter', 'latex', 'FontSize', 15);
zlabel('$x(k+1)$', 'Interpreter', 'latex', 'FontSize', 15);
legend('$95\%$ conf. bound','augmented func est', 'true func', '$x(k+1)$ train samples', ...
   'Location','southeast', 'Interpreter', 'latex', 'FontSize', 15);
