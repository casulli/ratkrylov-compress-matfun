% Compares low-memory polynomial Krylov methods for the computation of A^(-1/2), where A is a discretization of the 2D Laplacian
% Corresponds to Figure 2, Table 2 and Table 3 in the paper

addpath("utils");
addpath("utils-comparison");
clear;
maxNumCompThreads(1);

nn0 = 200*[1:5];

kk = zeros(1, length(nn0));
kk_aaa = zeros(1, length(nn0));
time = zeros(7, length(nn0));
err = zeros(7, length(nn0));
iter = zeros(7, length(nn0));

for j = 1:length(nn0)
	n0 = nn0(j);
	A = gallery('poisson', n0);
	A = A * ((n0+1)^2);
	n = size(A, 1);
	b = ones(n, 1);
	b = b/norm(b);
	nrmb = norm(b);
	f = @(x) inv(sqrtm(x));

	options.tol = 1e-8;
	options.isreal = true;
	maxit = 3000;
	infpoles = inf*ones(1, maxit);
	options.maxit = maxit;

	C = gallery('tridiag', n0);
	C = C * ((n0+1)^2);
	ev = eig(C);
	options.a = 2*min(abs(ev));
	options.b = 2*max(abs(ev));

	% Exact solution:
	I = speye(n0);
	[VC, DC] = eig(full(C));		% C = VC * DC * VC'
	DC = sparse(DC);
	c = ones(n0, 1);	c = c/norm(c);
	Ve = VC'*c;
	FVE = reshape(f(kron(DC, I) + kron(I, DC)) * kron(Ve, Ve), n0, n0);
	FAb = reshape(VC*FVE*VC', n0^2, 1);

	% RKcompress: 
	fprintf("RKcompress\n");
	tic;
	[y, iter(1,j), k, m] = RKcompress_fAb(A, b, "invsqrt", inf, f, options.tol, options);
	time(1,j) = toc;
	err(1,j) = norm(y-FAb)/norm(FAb);
	kk(j) = k;

	% Two-pass Lanczos:
	fprintf("twopass lanczos\n");
    options.checkconv = 1;
	tic;
	[y, iter(2,j)] = lanczos_fAb_twopass(A, b, infpoles, f, options.tol, options);
	time(2,j) = toc;
	err(2,j) = norm(y-FAb)/norm(FAb);

	% Multishift CG:
	fprintf("multishift CG\n");
	Z = chebpts(1000, [options.a, options.b]);
	F = 1./sqrt(Z);
	[r, pol, res, zer] = aaa(F, Z, 'tol', 1e-4*options.tol);
	kk_aaa(j) = length(pol);
	g = @(z) reshape(sum(res./(z(:)' - pol), 1), length(z), 1);
	c = sum(r(Z) - g(Z))/length(Z);			% constant term
	raterr = max(abs(r(Z) - g(Z) - c));		% error in partial fraction form
	removeflag = false;
	tic;
	[y, iter(3,j)] = multishiftCG_fAb(A, b, res, pol, options.tol, maxit, removeflag);
	y = y + c*b;		% constant term correction
	time(3,j) = toc;
	err(3,j) = norm(y-FAb)/norm(FAb);

	% Multishift CG with removal:
	fprintf("multishift CG with removal\n");
	removeflag = true;
	minremove = 1;
	tic;
	[y, iter(4,j)] = multishiftCG_fAb(A, b, res, pol, options.tol, maxit, removeflag, minremove);
	y = y + c*b;		% constant term correction
	time(4,j) = toc;
	err(4,j) = norm(y-FAb)/norm(FAb);

	% Stieltjes restarted:
	fprintf("restarted Stieltjes\n");
	param.function = 'invSqrt';
	param.restart_length = k+m;             % restart cycle length
	param.max_restarts = 300;               % number of restart cycles
	param.tol = options.tol;           % tolerance for quadrature rule
	param.transformation_parameter = 1;     % parameter for the integral transformation
	param.hermitian = 1;                    % set 0 if A is not Hermitian
	param.V_full = 0;                       % set 1 if you need Krylov basis
	param.H_full = 0;                       % do not store all Hessenberg matrices
	param.exact = [];
	param.stopping_accuracy = options.tol;  % stopping accuracy
	param.inner_product = @(a,b) b'*a;      % use standard euclidean inner product
	param.thick = [];                       % no implicit deflation is performed
	param.min_decay = 0.95;                 % we desire linear error reduction of rate < .95 
	param.waitbar = 0; 
	param.reorth_number = 0;                % reorthogonalizations
	param.truncation_length = inf;          % truncation length for Arnoldi
	param.verbose = 0;
	tic;
	[y, out1] = funm_quad(A, b, param);
	time(5,j) = toc;
	err(5,j) = norm(y-FAb)/norm(FAb);
	iter(5,j) = size(out1.appr, 2)*param.restart_length;

	% Stieltjes restarted with deflation
	fprintf("restarted Stieltjes with deflation\n")
	param.thick = @thick_quad;              % Thick restart function for implicit deflation
	param.number_thick = 5;                 % Number of target eigenvalues for implicit deflation
	param.exact = [];
	tic;
	[y, out2] = funm_quad(A, b, param);
	time(6,j) = toc;
	err(6,j) = norm(y-FAb)/norm(FAb);
	iter(6,j) = size(out2.appr, 2)*param.restart_length;

	% RKcompress with AAA poles: 
	% fprintf("RKcompress with AAA poles\n");
	% tic;
	% [y, iter(7,j), k, m] = RKcompress_fAb(A, b, pol, inf, f, options.tol, options);
	% time(7,j) = toc;
	% err(7,j) = norm(y-FAb)/norm(FAb);

    % Two-pass Lanczos20:
	fprintf("twopass lanczos20\n");
    options.checkconv = 20;
	tic;
	[y, iter(7,j)] = lanczos_fAb_twopass(A, b, infpoles, f, options.tol, options);
	time(7,j) = toc;
	err(7,j) = norm(y-FAb)/norm(FAb);


	fprintf("\n");
end

% plot data
dlmwrite('output-data/invsqrt_k.dat', [kk.', kk_aaa.'], '\t');
dlmwrite('output-data/invsqrt_low-memory.dat',[nn0.', time(1,:).'],'\t');
dlmwrite('output-data/invsqrt_lanczos-twopass.dat',[nn0.', time(2,:).'],'\t');
dlmwrite('output-data/invsqrt_multishiftCG.dat',[nn0.', time(3,:).'],'\t');
dlmwrite('output-data/invsqrt_multishiftCG-rem.dat',[nn0.', time(4,:).'],'\t');
dlmwrite('output-data/invsqrt_restarted.dat',[nn0.', time(5,:).'],'\t');
dlmwrite('output-data/invsqrt_restarted-defl.dat',[nn0.', time(6,:).'],'\t');
% dlmwrite('output-data/invsqrt1_low-memory-aaa.dat',[nn0.', time(7,:).'],'\t');
dlmwrite('output-data/invsqrt_lanczos-twopass20.dat',[nn0.', time(7,:).'],'\t');

% table data
for j = 1:length(nn0)
	err_str(j) = sprintf("%.2e & %.2e & %.2e & %.2e & %.2e & %.2e \\\\", err(1:end-1, j));
	iter_str(j) = sprintf("%d & %d & %d & %d & %d & %d \\\\", iter(1:end-1, j));
end
fid = fopen("output-data/invsqrt1_err_table.txt", 'w');
for j = 1:length(nn0)
	fprintf(fid, "%d & %s\n", nn0(j)^2, err_str(j));
end
fclose(fid);
fid2 = fopen("output-data/invsqrt1_iter_table.txt", 'w');
for j = 1:length(nn0)
	fprintf(fid2, "%d & %s\n", nn0(j)^2, iter_str(j));
end
fclose(fid2);
