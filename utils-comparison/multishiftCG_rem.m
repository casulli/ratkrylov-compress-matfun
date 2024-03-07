function [xfinal, it] = multishiftCG_rem(A, b, xi, tol, maxit, min_idx_converged)
	% [xfinal, it] = multishiftCG_rem(A, b, xi, tol, maxit, min_idx_converged)
	% Solves the linear systems (A - xi(j) I) x = b using multishift CG method with removal
	% See [1, Algorithm 5] and [2, Section 5.3] for details
	%
	% [1] A. Frommer, P. Maass, Fast CG-based methods for Tikhonov-Phillips regularization, SIAM J. Sci. Comput., 1999
	% [2] J. van den Eshof, A. Frommer, T. Lippert, K. Schilling, H. van der Vorst, Numerical methods for the QCDd 
	%     overlap operator. I. Sign-function and error bounds, Computer Physics Communications, 2002
	%
	% Input:
	%	A					matrix or function handle that computes matvecs with A
	%	b					vector
	%	xi					vector of shifts (must be all finite)
	% 	tol					relative stopping tolerance
	%	maxit 				maximum number of iterations for CG
	%	min_idx_converged	minimum number of converged linear systems to remove simultaneously 
	%						(for less frequent memory management; default = 1)
	%
	% Output:
	%	xfinal		final approximate solutions
	%	it			number of iterations

	if nargin < 5
		maxit = length(b);
	end
	if nargin < 6
		min_idx_converged = 1;
	end
	if ~isa(A,'function_handle')
		A = @(v) A*v;
	end
	xi = xi(:).';				% row vector
	k = length(xi);
	n = length(b);

	% Initialize Lanczos iteration:
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
	it = ones(1, k);			% iteration counts
	atol = tol*nb;				% absolute tolerance
	idx_left = 1:k;				% indices of non converged systems
	idx_left_absolute = 1:k;	% indices with respect to starting indexing
	xfinal = zeros(n, k);

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

		it(idx_left_absolute) = it(idx_left_absolute)+1;	% update iteration count
		% Check convergence and remove converged systems:
		idx_converged = find(abs(sigma) < atol);
		% if ~isempty(idx_converged)
		if length(idx_converged) >= min_idx_converged	
			% Remove converged systems:
			idx_left = find(abs(sigma) >= atol);
			idx_converged_absolute = idx_left_absolute(idx_converged);
			xfinal(:, idx_converged_absolute) = x(:, idx_converged);
			
			idx_left_absolute = idx_left_absolute(idx_left);
			if isempty(idx_left)
				return;		% all systems converged
			end
			xi = xi(idx_left);
			gamma = gamma(idx_left);
			omega = omega(idx_left);
			sigma = sigma(idx_left);
			x = x(:, idx_left);
			p = p(:, idx_left);

			min_idx_converged = min(min_idx_converged, length(idx_left));
		end
	end
	xfinal(:, idx_left_absolute) = x;
end