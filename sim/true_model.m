function [h, dh, h_expr, dh_expr] = true_model(x_eval, u_eval, noise)
%nominal_model - Nominal model evaluation and its gradient on
% a deterministic/stochastic input omega. The nominal model is:
%   f(x(k), u(k))
syms x u

f_nominal = @(x, u)(0.1*x + tanh(x) + u);
f_expr = sym(f_nominal);

g_func = @(x,u)(sin(3*x) + u);
g_expr = sym(g_func);

h_expr = f_expr + g_expr;

dh_expr = jacobian(h_expr, [x, u]);

x = x_eval;
u = u_eval;

% Evaluate nominal model
h = double(vpa(subs(h_expr))) + noise;
% Evaluate Jacobian
dh = double(vpa(subs(dh_expr)));
end
