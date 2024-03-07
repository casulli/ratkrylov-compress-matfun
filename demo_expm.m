% Compares polynomial Krylov methods for the computation of exp(-tA), where A is a discretization of the 2D Laplacian
% Corresponds to Figure 1 and Table 1 in the paper

addpath("utils");
addpath("utils-comparison");
clear;
maxNumCompThreads(1);

n0 = 1000;
A = -gallery('poisson', n0);
n = size(A,1);
A = A * ((n0+1)^2);
tt = [1e-5 1e-4 1e-3 1e-2 1e-1];
b = ones(n, 1);
nrmb = norm(b);

options.tol = 1e-10;
options.isreal = true;
maxit = 2000;
options.maxit = maxit;
infpoles = inf*ones(1, maxit);

time = zeros(4, length(tt));
err = zeros(4, length(tt));
iter = zeros(4, length(tt));

for j = 1:length(tt)
	t = tt(j);
	fprintf("%.4e\n", t);
	B = t*A;

	% Exact solution:
	C = -gallery("tridiag", n0);
	C = C * ((n0+1)^2);
    FC = expm(t*C)*ones(n0,1);
	FAb = kron(FC, FC);

	% RKcompress: 
	fprintf("RKcompress\n");
	tic;
	[y, iter(1, j)] = RKcompress_fAb(B, b, "exp", inf, @expm,options.tol, options);
	time(1, j) = toc;
	err(1, j) = norm(y-FAb)/norm(FAb);

	% Lanczos:
	fprintf("lanczos\n");
	tic; 
	[y, iter(2, j)] = lanczos_fAb(B, b, infpoles, @expm, options.tol, options);
	time(2, j) = toc;
	err(2, j) = norm(y-FAb)/norm(FAb);

	% Two-pass Lanczos:
	fprintf("twopass lanczos\n");
	options.fast2pass = true;
	tic;
	[y, iter(3, j)] = lanczos_fAb_twopass(B, b, infpoles, @expm, options.tol, options);
	time(3, j) = toc;
	err(3, j) = norm(y-FAb)/norm(FAb);

	% Full-orthogonalization Arnoldi:
	if (j <= 3)
		fprintf("full arnoldi\n");
		s = 1;
		m = 1;
		tic;
		[V, H] = arnoldi(B, b, [], s);
		haty = expm(H(1:s, 1:s))*eye(s, 1);
		% haty = expm(H(1:s,1:s)/K(1:s,1:s))*eye(s,1);
		haty_old = zeros(0,1);
		errest = norm(haty - [haty_old; zeros(s,1)]);
		while (errest > options.tol * norm(haty) && s < maxit)
			haty_old = haty;
			[V, H] = arnoldi(B, V, H, m);
			s = s + m;
			haty = expm(H(1:s, 1:s))*eye(s, 1);
			% haty = expm(H(1:s,1:s)/K(1:s,1:s)) * eye(s,1);
			errest = norm(haty - [haty_old; zeros(m,1)]);
		end
		y = V(:,1:s)*haty*nrmb;
		time(4, j) = toc;
		iter(4, j) = s;
		err(4, j) = norm(y-FAb)/norm(FAb);
	end
	fprintf("\n");
	
end

% check iter and error discrepancy among different methods
iter_check = iter - iter(2, :);		% should be 0
err_check = err - err(2, :);		% should be close to 0

% plot data
dlmwrite('output-data/exp_low-mem.dat',[tt.',time(1,:).'],'\t');
dlmwrite('output-data/exp_Lanczos.dat',[tt.',time(2,:).'],'\t');
dlmwrite('output-data/exp_Lanczos-twoPass.dat',[tt.',time(3,:).'],'\t');
dlmwrite('output-data/exp_Arnoldi-full.dat',[tt(1:3).',time(4,1:3).'],'\t');

% table data
iter_str = sprintf("iter & & %d & %d & %d & %d & %d \\\\", iter(2, :));
err_str = sprintf("err & & %.2e & %.2e & %.2e & %.2e & %.2e \\\\", err(2, :));
fid = fopen("output-data/exp_table.txt", 'w');
fprintf(fid, "%s\n%s", iter_str, err_str);
fclose(fid);




