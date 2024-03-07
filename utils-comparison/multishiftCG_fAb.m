function [y, iter] = multishiftCG_fAb(A, b, alpha, theta, tol, maxit, removeflag, minremove)
	% [y, iter] = multishiftCG_fAb(A, b, alpha, theta, tol, maxit, removeflag, minremove)
	% Approximates f(A)*b by applying multishift CG to a rational approximation of f(A) written in partial fraction form
	% 	r(z) = alpha(1)/(z - theta(1)) + ... + alpha(k)/(z - thheta(k))
	%	f(A)*b = alpha(1)*(A - theta(1))\b + ... + alpha(k)*(A - theta(k))\b
	% Stopping criterion: assuming that
	%	r is such that || f - r || < || f || * tol/2 on W(A)
	%	and all linear system residuals are such that 
	%		|| residual || < tol/2 * || b ||  / (1 + tol/2),
	% 	then the final error satisfies:
	%		|| f(A)*b - y || < tol * || f || * || b ||
	%
	% Input:
	%	A			matrix or function handle that computes matvecs with A
	%	b			vector
	%	alpha		coefficients of the rational function in partial fraction form
	%	theta		poles of the rational function (must be all finite)
	% 	tol			relative stopping tolerance
	%	maxit 		maximum number of iterations for CG
	%	removeflag	boolean; if true, remove converged linear systems for increased efficiency (default = false)
	%	minremove	minimum number of converged linear systems required to remove them (default = 1)
	%
	% Output:
	%	y			final approximate solution
	%	iter		number of iterations

	if nargin < 6
		maxit = length(b);
	end
	if nargin < 7
		removeflag = false;
	end
	if nargin < 8
		minremove = 1;
	end
	if ~isa(A,'function_handle')
		A = @(v) A*v;
	end

	% Run multishift CG:
	tolcg = tol/2 / (1+tol/2);		% relative tolerance for CG
	% tolcg = tol;
	if removeflag
		[x, iter] = multishiftCG_rem(A, b, theta, tolcg, maxit, minremove);
		else
		[x, iter] = multishiftCG(A, b, theta, tolcg, maxit);
	end
	% Reconstruct approximation to f(A)*b:
	y = x*alpha(:);
	iter = max(iter);
end