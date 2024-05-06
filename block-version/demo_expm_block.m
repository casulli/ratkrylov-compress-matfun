% Compares polynomial Krylov methods for the computation of exp(-tA), where A is a discretization of the 2D Laplacian
% Corresponds to Figure 1 and Table 1 in the paper

addpath("utils");
%addpath("utils-comparison");
clear;
maxNumCompThreads(1);

n0 = 500;
A = -gallery('poisson', n0);
n = size(A,1);
A = A * ((n0+1)^2);
tt = [1e-5 1e-4 1e-3 1e-2 1e-1];
rng(1);
b1 = randn(n0, 2);
b2 = randn(n0, 2);
b = kron(b1,b2);
bs = size(b,2);

options.tol = 1e-10;
options.isreal = true;
maxit = 1700;
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
    FC = expm(t*C);
	FAb = kron(FC*b1, FC*b2);
 
	% RKcompress: 
	fprintf("RKcompress\n");
	tic;
	[y, iter(1, j)] = RKcompress_fAb_block(B, b, "exp", @expm,options.tol, options);
	time(1, j) = toc;
	err(1, j) = norm(y-FAb,"fro")/norm(FAb,"fro");

	% Lanczos:
	fprintf("lanczos\n");
	tic; 
	[y, iter(2, j)] = lanczos_fAb_block(B, b, @expm, maxit, options.tol);
	time(2, j) = toc;
	err(2, j) = norm(y-FAb,"fro" )/norm(FAb,"fro");

	% Two-pass Lanczos:
	fprintf("twopass lanczos\n");
	options.fast2pass = true;
	tic;
    [y, iter(3, j)] = lanczos_fAb_twopass_block(B, b, @expm, maxit, options.tol, options);
	time(3, j) = toc;
	err(3, j) = norm(y-FAb,"fro")/norm(FAb,"fro");

	% Full-orthogonalization Arnoldi:
	if (j <= 3)
		fprintf("full arnoldi\n");
		s = 1;
		m = 1;
		tic;
        [V,R] = qr(b,0);
        [V, H] = arnoldi_block(B, V, [], s, bs);
		haty = expm(H(1:s*bs, 1:s*bs))*eye(s*bs, bs);
        haty_old = zeros(0,bs);
		errest = norm(haty - [haty_old; zeros(s*bs,bs)],"fro");
		while (errest >= options.tol * norm(haty,"fro") && s < maxit)
			haty_old = haty;
			[V, H] = arnoldi_block(B, V, H, m, bs);
			s = s + m;
			haty = expm(H(1:s*bs, 1:s*bs))*eye(s*bs, bs);
			errest = norm(haty - [haty_old; zeros(m*bs,bs)],"fro");
		end
		y = V(:,1:s*bs)*haty*R;
		time(4, j) = toc;
		iter(4, j) = s;
		err(4, j) = norm(y-FAb,"fro")/norm(FAb,"fro");
	end
	fprintf("\n");
	
end

% check iter and error discrepancy among different methods
iter_check = iter - iter(2, :);		% should be 0
err_check = err - err(2, :);		% should be close to 0

% plot data
dlmwrite('output-data/exp_low-mem_block.dat',[tt.',time(1,:).'],'\t');
dlmwrite('output-data/exp_Lanczos_block.dat',[tt.',time(2,:).'],'\t');
dlmwrite('output-data/exp_Lanczos-twoPass_block.dat',[tt.',time(3,:).'],'\t');
dlmwrite('output-data/exp_Arnoldi-full_block.dat',[tt(1:3).',time(4,1:3).'],'\t');

% table data
iter_str = sprintf("iter & & %d & %d & %d & %d & %d \\\\", iter(2, :));
err_str = sprintf("err & & %.2e & %.2e & %.2e & %.2e & %.2e \\\\", err(2, :));
fid = fopen("output-data/exp_table_block.txt", 'w');
fprintf(fid, "%s\n%s", iter_str, err_str);
fclose(fid);




