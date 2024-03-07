function [y, iter, k, m, errest] = RKcompress_fAb(A, b, xi, theta, f, tol, options)
	% function [y, iter, k, m, errest] = RKcompress_fAb(A, b, xi, theta, f, tol, options)
	% Computes f(A)b for A Hermitian, employing the low-memory rational Lanczos algorithm 
	% with rational Krylov compression described in [1]
	% The algorithm checks convergence by monitoring the difference between iterations 
	% whenever an outer pole (i.e. theta(i)) equals to infinity and after each compression
	%
	% [1] A. A. Casulli , I. Simunec, A low-memory Lanczos method with rational Krylov compression 
	%     for matrix functions, arxiv preprint, 2024
	%
	% Input:
	%	A			can be a full/sparse matrix or a function handle that performs the
	%   			matrix vector product v -> A*v
	%				if A is a function handle, 
	%	b			vector
	%	xi			vector that contains the poles for the inner Krylov subspaces
	% 				The following built-in choices are available:
	%					xi = "exp" computes the poles for the exponential for A with 
	%						spectrum in	(-inf, 0]
	%					xi = "invsqrt" computes poles for the gamma-th power with -1<gamma<0
	%       				in such a case an interval [options.a, options.b] with options.a > 0 that
	%      					contains the eigenvalues of A must be provided
	% 					xi = "markov" computes the poles for a generic Markov function
	% 						in such a case, the following fields are required: 
	%       					options.a = lower bound for the eigenvalues of A
	%       					options.b = upper bound for the eigenvalues of A
	%       					options.alpha, options.beta extrema in the integral representation of f
	%	theta		vector that contains the outer poles that are cyclically used in the procedure
	%				for instance, for polynomial Krylov methods theta = inf 
	%				or for shift-and-invert theta = pol_SI
	%	f			matrix valued function handle (e.g., @expm for the exponential function)
	%	tol			requested relative accuracy
	%	options		struct with fields:
	%		options.m						number of steps after which recompression occurs (default = length(xi))
	%		options.maxit					maximum number of outer iterations
	%		options.a, options.b			extrema of the spectral interval of A
	%		options.alpha, options.beta		extrema in the integral representation of a Markov function f
	%		options.solveSystems			cell(1, length(theta)) such that:
	%   		options.solveSystems{i} is a function handle that solves the linear system 
	%			v -> (A-theta(i)I)\v	(if theta(i) = inf it contains @(v) v)
	%			by default, if A is a matrix this is done using the MATLAB command "decomposition"
	%		options.isreal					boolean; set to true if A is real 
	%										(option used in rational_krylov.m for basis compression)
	%
	% Output:
	%	y			final approximate solution
	%	iter		number of iterations
	%	k			number of poles used for inner rational Krylov subspace
	%	m			number of steps between recompressions
	%	errest		matrix with two columns, each row contains an iteration number corresponding 
	%				to an infinite pole and the corresponding error estimate

	if nargin < 7
		options = [];
	end

	options.tol = tol;

	if ~isa(A,'function_handle')
		mult = @(v) A*v;
	else 
		mult = A;
	end

	% preprocessing: factorizations for solving linear systems
	decomp = cell(1,length(theta));
	if isfield (options, 'solveSystems') == 0
		for i=1 : length(theta)
			check = false;
			for j = 1: i-1
				if theta (i) == theta (j)
					options.solveSystems{i} = options.solveSystems{j};
					check = true;
					break
				end
			end
			if ~check 
				if theta(i) == inf
						options.solveSystems{i} = @(v) v;
				else
					decomp{i} = decomposition(A - theta(i)*speye(size(A)));
					options.solveSystems{i} = @(v) decomp{i}\v;
				end
			end
		end
	end
	% ---------------------------------

	% built-in poles
	if (isstring(xi) || ischar(xi)) && xi == "invsqrt"
		options.alpha = -inf;
		options.beta = 0;
		xi = "markov";
	end


	if (isstring(xi) || ischar(xi)) && xi == "markov"
		if isfield (options, 'alpha')==0 || isfield (options, 'beta')==0
		error('the extrema in the integral representation of f are needed as options.alpha and options.beta') 
		end
		if isfield (options, 'a')==0 || isfield (options, 'b')==0
			error('An interval enclosing the eigenvalues of A must be provided.')
		else
			k = ceil(log(4/options.tol)*log(16*(options.b-options.beta)/(options.a-options.beta))/pi^2);
			xi_MF = poles_Markov_functions(options.a,options.b,options.alpha,options.beta,k);
			[y, iter, k, m, errest] = RKcompress_matfun(mult, b, xi_MF, theta, f, options);
		end
	elseif (isstring(xi) || ischar(xi)) && xi == "exp" % 25 poles for rational approximation of the exponential on the negative real axis
		xi_exp = [- 9.9998213693812510539248035448416 - 15.616962406068104596656675113336i,...
		- 9.9998213693812510539248035448416 + 15.616962406068104596656675113336i,...
		- 6.4919829900021201329203401765263 - 13.941894334714035916740557581743i,...
		- 6.4919829900021201329203401765263 + 13.941894334714035916740557581743i,...
		- 3.9903568233545400544998688881409 - 12.488253479931032049378079791714i,...
		- 3.9903568233545400544998688881409 + 12.488253479931032049378079791714i,...
		- 1.9861233615362475604899520854603 + 11.061961911284359810439320793404i,...
		- 1.9861233615362475604899520854603 - 11.061961911284359810439320793404i,...
		- 0.37795274103508612215755372673015 - 9.8028554590764619340119184986009i,...
		- 0.37795274103508612215755372673015 + 9.8028554590764619340119184986009i,...
		0.025911602037835421490493295886546 - 1.9400954320806222916523198594056i,...
		0.025911602037835421490493295886546 + 1.9400954320806222916523198594056i,...
		0.22666571234589860402019631242325 + 6.4351035714386513579805492329403i, ...
		0.22666571234589860402019631242325 - 6.4351035714386513579805492329403i,...
		1.2790664138698104753628831022021 - 8.8442696429749687974365643467319i,...
		1.2790664138698104753628831022021 + 8.8442696429749687974365643467319i,...
		3.217809737533906016046453213333 - 7.6770611136562451038079164817507i,...
		3.217809737533906016046453213333 + 7.6770611136562451038079164817507i,...
		4.9226944402749036334442444790599 + 6.1534807619405512756435860745859i,...
		4.9226944402749036334442444790599 - 6.1534807619405512756435860745859i,...
		6.218632182052133457085535773616 - 4.3009206470565567366228484528267i,...
		6.218632182052133457085535773616 + 4.3009206470565567366228484528267i,...
		7.0288588353989918272529822270328 - 2.211649636066695374619041437372i,...
		7.0288588353989918272529822270328 + 2.211649636066695374619041437372i, ...
		7.3045847287025542613021864714592];
		[y, iter, k, m, errest] = RKcompress_matfun(mult, b, xi_exp, theta, f, options);
	else
		[y, iter, k, m, errest] = RKcompress_matfun(mult, b, xi, theta, f, options);
	end	
	l = find(errest(:, 2) == 0, 1, 'first');
	if ~isempty(l)
		errest = errest(1:l-1, :);
	end

end