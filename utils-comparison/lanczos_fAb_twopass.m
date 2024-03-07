function [y, iter, errhist] = lanczos_fAb_twopass(A, b, theta, f, tol, options)
	% [y, iter, errhist] = lanczos_fAb_twopass(A, b, theta, f, tol, options)
	% Implementation of rational Lanczos for f(A)*b with two-pass strategy
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
	%		options.fast2pass		boolean; if true, use fast second pass implementation that does not recompute orthogonalization coefficients (default = true)
	%
	% Output:
	%	y			final approximate solution
	%	iter		number of iterations
	%	errhist		error norm history (estimates)

	if ~isa(A,'function_handle')
		mult = @(v) A*v;
	end
	k = length(theta);
	n = length(b);
	if nargin < 6
		options = [];
	end
	if isfield(options, 'fast2pass') == 0
		options.fast2pass = true;
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
	
	% First pass:
	W = zeros(n, 3);		% contains only last three columns of the Krylov basis
	alpha = zeros(1, k+1);
	beta = zeros(1, k+1);
	invpoles = zeros(1, k+1);
	invpoles(1) = 0;		% first pole is always infinity since b is included
	j_old = 0;
	yhat_old = [];

	for j = 1:k
		if j == 1
			nrmb = norm(b);	
			W(:, 2) = b/nrmb;
			[W(:, 3), alpha(1), beta(1)] = short_recurrence_Arnoldi_in(mult, options.solveSystems{1}, zeros(n,1), W(:,2), theta(1), inf, inf, 0);
		elseif j == 2
			[W(:, 3), alpha(2), beta(2)] = short_recurrence_Arnoldi_in(mult, options.solveSystems{2}, W(:,1), W(:,2), theta(2), inf, theta(1), beta(1));
		else
			[W(:, 3), alpha(j), beta(j)] = short_recurrence_Arnoldi_in(mult, options.solveSystems{j}, W(:,1), W(:,2), theta(j), theta(j-2), theta(j-1), beta(j-1));
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
			iter = j;
			% Check convergence:
			errest = norm(yhat - [yhat_old; zeros(j - j_old, 1)]);
			if nargout == 3
				errhist(end+1, :) = [j, errest/norm(yhat)];
			end
			if errest < tol*norm(yhat)
				% Convergence reached: start second pass to compute solution
				iter = j;
				break;
			else
				yhat_old = yhat;
				j_old = j;
			end
		end
		W(:, 1:2) = W(:, 2:3);		% discard old basis vector
	end

	if exist('yhat', 'var') == 0
		iter = k;
		y = NaN*zeros(n, 1);	% solution was never computed...
		warning('no infinite poles were used in rational Lanczos, cannot check convergence');
		return;
	end
	iter = length(yhat);

	% Second pass:
	if options.fast2pass
		% Faster implementation:
		y = two_pass_reconstruction_rational(mult, options.solveSystems, b, alpha(1:iter), beta(1:iter), theta(1:iter), yhat);
	else
		% Basic implementation that recomputes everything:
		y = zeros(n, 1);
		for j = 1:iter-1
			% Recompute basis:
			if j == 1
				nrmb = norm(b);	
				W(:, 2) = b/nrmb;
				y = y + W(:, 2)*yhat(1);
				[W(:, 3), alpha(1), beta(1)] = short_recurrence_Arnoldi_in(mult, options.solveSystems{1}, zeros(n,1), W(:,2), theta(1), inf, inf, 0);
			elseif j == 2
				[W(:, 3), alpha(2), beta(2)] = short_recurrence_Arnoldi_in(mult, options.solveSystems{2}, W(:,1), W(:,2), theta(2), inf, theta(1), beta(1));
			else
				[W(:, 3), alpha(j), beta(j)] = short_recurrence_Arnoldi_in(mult, options.solveSystems{j}, W(:,1), W(:,2), theta(j), theta(j-2), theta(j-1), beta(j-1));
			end
			y = y + W(:, 3)*yhat(j+1);	% update solution
			W(:, 1:2) = W(:, 2:3);		% discard old basis vector
		end	
	end

end 