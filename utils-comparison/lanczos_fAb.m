function [y, iter, errhist] = lanczos_fAb(A, b, theta, f, tol, options)
	% [y, iter, errhist] = lanczos_fAb(A, b, theta, f, tol, options)
	% Implementation of rational Lanczos for f(A)*b
	% the solution is only computed in iterations with an infinite pole
	% (in particular, no solution is computed if no infinite poles are used)
	%
	% Input:
	%	A			matrix or function handle that computes matvecs with A
	%	b			vector
	%	theta		poles for rational Lanczos; the starting vector b is always included in the Krylov space
	%	f			function; we assume that f can compute matrix arguments
	% 	tol			relative stopping tolerance
	%	options		struct with fields:
	%		options.maxit			maximum number of iterations
	%		options.solveSystems	cell array of function handles, such that solveSystems{i}(v) = (A - theta(i)*I)^(-1)*b 
	%				(can be populated automatically if A is a matrix; must be provided as input if A is a function handle)
	%
	% Output:
	%	y			final approximate solution
	%	iter		number of iterations
	%	errhist		error norm history (estimates)

	if ~isa(A, 'function_handle')
		mult = @(v) A*v;
	end
	k = length(theta);
	if isfield(options, "maxit")
		k = min(k, options.maxit);
	end
	n = length(b);
	if nargin < 6
		options = [];
	end
	if nargout == 3
		errhist = [];
	end

	% preprocessing: factorizations for solving linear systems
	decomp = cell(1, k);
	if isfield (options, 'solveSystems') == 0
		for i=1 : k
			check = false;
			for j = 1: i-1
				if theta (i) == theta (j)
					options.solveSystems{i} = options.solveSystems{j};
					check = true;
					break;
				end
			end
			if ~check 
				if theta(i) == inf
						options.solveSystems{i} = @(v) v;
				else
					if isa(A, 'function_handle')
						error("options.solveSystems must be given as input if A is a function handle");
					end
					decomp{i} = decomposition(A - theta(i)*speye(size(A)));
					options.solveSystems{i} = @(v) decomp{i}\v;
				end
			end
		end
	end
	% Shift matrix if necessary:
	if min(abs(theta)) <= tol
		mu = + max(theta(theta~=inf)) + 1;
		mult = @(v) mult(v) - mu*v;
		theta = theta - mu;
	else
		mu = 0;
	end

	% Run iterations of rational Lanczos:
	W = zeros(n, k+1);
	alpha = zeros(1, k+1);
	beta = zeros(1, k+1);
	invpoles = zeros(1, k+1);
	invpoles(1) = 0;		% first pole is always infinity since b is included
	j_old = 0;
	yhat_old = [];

	for j = 1:k
		if j == 1
			nrmb = norm(b);	
			W(:, 1) = b/nrmb;
			[W(:, 2), alpha(1), beta(1)] = short_recurrence_Arnoldi_in(mult, options.solveSystems{1}, zeros(n,1), W(:,1), theta(1), inf, inf, 0);		
		elseif j == 2
			[W(:, 3), alpha(2), beta(2)] = short_recurrence_Arnoldi_in(mult, options.solveSystems{2}, W(:,1), W(:,2), theta(2), inf, theta(1), beta(1));
		else
			[W(:, j+1), alpha(j), beta(j)] = short_recurrence_Arnoldi_in(mult, options.solveSystems{j}, W(:,j-1), W(:,j), theta(j), theta(j-2), theta(j-1), beta(j-1));
		end
		if theta(j) == inf
			invpoles(j+1) = 0;
		else
			invpoles(j+1) = 1/theta(j);
		end
		% Compute solution and check convergence if last pole is infinity:
		if theta(j) == inf
			H = diag(alpha(1:j)) + diag(beta(1:j-1),-1) + diag(beta(1:j-1),-1)';
			K = eye(j) + diag(invpoles(1:j)) * H;
			T = H/K + mu*eye(j);
			yhat = f(T)*[nrmb; zeros(j-1, 1)];
			% Check convergence:
			errest = norm(yhat - [yhat_old; zeros(j - j_old, 1)]);
			if nargout == 3
				errhist(end+1, :) = [j, errest];
			end
			if errest < tol*norm(yhat)
				y = W(:, 1:j)*yhat;
				iter = j;
				return;
			else
				yhat_old = yhat;
				j_old = j;
			end
		end
	end
	iter = k;
	if ~exist('y', 'var')
		if ~exist('j_old', 'var')
			warning("no infinite poles used: cannot compute solution");
			y = NaN*zeros(n, 1);
			return
		else
			y = W(:, 1:j_old)*yhat;
		end
	end

end 