function [x, it] = multishiftCG(A, b, xi, tol, maxit)
	% [x, it] = multishiftCG(A, b, xi, tol, maxit)
	% Solves the linear systems (A - xi(j) I) x = b using multishift CG method
	% See [1, Algorithm 5] for details
	%
	% [1] A. Frommer, P. Maass, Fast CG-based methods for Tikhonov-Phillips regularization, SIAM J. Sci. Comput., 1999
	%
	% Input:
	%	A			matrix or function handle that computes matvecs with A
	%	b			vector
	%	xi			vector of shifts (must be all finite)
	% 	tol			relative stopping tolerance
	%	maxit 		maximum number of iterations for CG
	%
	% Output:
	%	x			final approximate solution
	%	it			number of iterations
	
	if nargin < 5
		maxit = length(b);
	end
	if ~isa(A,'function_handle')
		A = @(v) A*v;
	end
	xi = xi(:).';				% row vector
	k = length(xi);
	n = length(b);
	nb = norm(b);
	beta = nb;
	v = b/beta;					% current basis vector
	v0 = zeros(n,1);			% old basis vector
	% Initialize quantities for CG:
	x = zeros(n, k);			% solutions
	p = b.*ones(1, k);
	gamma = ones(1, k);
	sigma = beta*ones(1, k);	% residual norms
	omega = zeros(1, k);
	atol = tol*nb;				% absolute tolerance

	for j = 2:maxit
		% Lanczos iteration:
		Av = A(v);
		delta = v'*Av;
		w = Av - delta*v - beta*v0;
		beta = norm(w);
		v0 = v;
		v = w/beta;

		% Update solutions of shifted linear system:
		gamma = 1./(delta - xi - omega./gamma);
		omega = (beta*gamma).^2;
		sigma = -beta*(gamma.*sigma);
		x = x + p.*gamma;			% update solutions
		p = v.*sigma + p.*omega;
		% Check convergence:
		if max(abs(sigma)) < atol
			it = j;
			return;				% Convergence reached for all systems
		end
	end
	it = maxit;

end