function [mean, cov] = taylor_approx(omega, omega_star, post, hyp, covfunc)
%TAYLOR_APPROX Summary of this function goes here
%   Detailed explanation goes here

omega_star_mean = omega_star{1};
omega_star_cov = omega_star{2};

[z_mu_, z_var_, alpha, Ks, gp_jacobian_tools] = gp_model(omega, omega_star_mean, post, hyp, covfunc);

tp_expr = gp_jacobian_tools{1};
Ks_expr = gp_jacobian_tools{2};
dKs_expr = gp_jacobian_tools{3};
dKs = double(vpa(subs(subs(dKs_expr, tp_expr, omega_star_mean), Ks_expr, Ks)));
jacobian_gp_mean_val = dKs'*alpha;
jacobian_gp_mean_val = jacobian_gp_mean_val';

[f_val, jacobian_f_val] = nominal_model(omega_star_mean);

mean = f_val + z_mu_;

A = [jacobian_f_val eye(size(jacobian_f_val,1))];
cov_f_z_tilde = omega_star_cov*jacobian_gp_mean_val';
cov_z_tilde = z_var_ + jacobian_gp_mean_val*omega_star_cov*jacobian_gp_mean_val';
cov = A*[omega_star_cov cov_f_z_tilde; cov_f_z_tilde' cov_z_tilde]*A';

end
