function [check, T, Tcorr, w, z_conv2, relerrest] = check_convergence(alpha, beta,invPoles, T, Tcorr, w, f, c, z_conv, tol, maxnrm, mu)
	% Checks convergence and updates T within RKcompress

	m = length(alpha);
	k = size(T,1);
	H = diag(alpha)+diag(beta(1:(m-1)),1)+diag(beta(1:(m-1)),-1);
	K = eye(size(H)) + diag(invPoles)*H;
	T = blkdiag(T, H/K + mu*eye(size(H,1)));
	T(k+1,k+1) = T(k+1,k+1) + Tcorr;		% Tcorr is computed before the end of previous iteration
	T(1:k,k+1) = w;
	T(k+1,1:k) = w';
	Tcorr = - beta(m)^2 * invPoles(m) * [zeros(m-1, 1); 1]'*(K\[zeros(m-1, 1); 1]);
	w = [zeros(1,k), [zeros(1,m-1), beta(m)] / K]';
	z_conv2 = f(T)*c;
	relerrest = norm(z_conv2 - [z_conv ; zeros(m,1)]) / max(norm(z_conv2),maxnrm);
	if (relerrest < tol)
		check = true;
	else
		check = false;
	end
end