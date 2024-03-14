function [f, df, f_expr, df_expr] = nominal_model(x_eval, u_eval)
%nominal_model - Nominal model evaluation and its gradient on
% a deterministic/stochastic input omega. The nominal model is:
%   f(x(k), u(k))
syms x u

f_nominal = @(x, u)(0.1*x + tanh(x) + u);
f_expr = sym(f_nominal);
df_expr = jacobian(f_expr, [x, u]);

x = x_eval;
u = u_eval;

% Evaluate nominal model
f = double(vpa(subs(f_expr)));
% Evaluate Jacobian
df = double(vpa(subs(df_expr)));
end
