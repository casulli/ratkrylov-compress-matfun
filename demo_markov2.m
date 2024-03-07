% Compares shift-and-invert Krylov methods for the computation of A^(-1/2), where A is a discretization of the 1D Laplacian
% Corresponds to Table 4 in the paper

addpath("utils");
addpath("utils-comparison");
clear;
maxNumCompThreads(1);

nn = [50000 100000 150000 200000];

time = zeros(2, length(nn));
err = zeros(2, length(nn));
iter = zeros(2, length(nn));

for j = 1:length(nn)
	n = nn(j);
	A = gallery('tridiag', n);
	A = A * (n+1)^2;
	b = ones(n, 1);
	b = b / norm(b);
	nrmb = norm(b);
	f = @(x) inv(sqrtm(x));

	options.a = eigs(A, 1, 'smallestabs');
	options.b = 2*max(abs(diag(A)));
	options.maxf = abs(f(options.a));
	options.alpha = -inf;
	options.beta = 0;

	options.isreal = true;
	options.tol = 1e-6;
	maxit = 1000;
	options.maxit = maxit;
	xi_SI = polesingle_Markov_functions(options.a, options.b, options.alpha, options.beta);

	% Exact solution with rational Krylov:
	rktol = 1e-8;
	rkopts.isreal = true;
	k = ceil(log(2*options.maxf/rktol)*log(16*(options.b-options.beta)/(options.a-options.beta))/pi^2);
	xi_MF = poles_Markov_functions(options.a, options.b, options.alpha, options.beta, k);
	V = rational_krylov(A, b, xi_MF, rkopts);
	T = V'*(A*V);
	haty = f(T)*(V'*b);
	FAb = V * haty;

	% [Vrk, K, H] = rat_krylov(A, b, xi_MF);
	% Trk = Vrk'*(A*Vrk);
	% hatyrk = f(Trk)*eye(size(Vrk, 2), 1);
	% FAbrk = Vrk * (hatyrk*nrmb);

	% RK solution check:
	xi_MF2 = poles_Markov_functions(options.a, options.b, options.alpha, options.beta, floor(2*k));
	V2 = rational_krylov(A, b, xi_MF2, rkopts);
	T2 = V2'*(A*V2);
	haty2 = f(T2)*(V2'*b);
	FAb2 = V2 * haty2;
	
	% [V2rk, K2, H2] = rat_krylov(A, b, xi_MF2);
	% T2rk = V2rk'*(A*V2rk);
	% haty2rk = f(T2rk)*eye(size(V2rk, 2), 1);
	% FAb2rk = V2rk * (haty2rk*nrmb);

	rkerrest = norm(FAb - FAb2)/norm(FAb);
	fprintf("%.4e\n", rkerrest)


	% RKcompress, shift-and-invert: 
	fprintf("RKcompress, shift-and-invert\n");
	options.m = 10;
	tic;
	[y, iter(1, j), k, m, errhist1] = RKcompress_fAb(A, b, "invsqrt", xi_SI, f, options.tol, options);
	time(1, j) = toc;
	err(1, j) = norm(y-FAb)/norm(FAb);

	% Two-pass shift-and-invert:
	fprintf("twopass shift-and-invert\n");
	pol0 = [xi_SI*ones(1, k-1), kron(ones(1, 100), [Inf, xi_SI*ones(1, m-1)])];
	options.fast2pass = true;
	tic;
	[y, iter(2, j), errhist2] = lanczos_fAb_twopass(A, b, pol0, f, options.tol, options);
	time(2, j) = toc;
	err(2, j) = norm(y-FAb)/norm(FAb);
end

% table data
fid = fopen("output-data/si_table.txt", 'w');
for j = 1:length(nn)
	fprintf(fid, "%d & %.2f & %.2f & %.2e & %.2e & %d \\\\\n", nn(j), time(1, j), time(2, j), err(1, j), err(2, j), iter(1, j));
end	
fclose(fid);