function [y, iter, k, m, errest] = RKcompress_matfun(mult, b, xi, theta, f, options)
	% [y, iter, k, m, errest] = RKcompress_matfun(mult, b, xi, theta, f, options)
	% Computes f(A)b for A Hermitian, employing the low-memory rational Lanczos algorithm 
	% with rational Krylov compression described in [1]
	%
	% [1] A. A. Casulli , I. Simunec, A low-memory Lanczos method with rational Krylov compression 
	%     for matrix functions, arxiv preprint, 2024
	%
	% see RKcompress_fAb.m for input and output details


	n = size(b,1);
	k = length(xi);

	if nargin < 6
		options = [];
	end
	if isfield (options, 'maxit') == 0
		options.maxit = n;
	end
	if isfield (options, 'm') == 0
		m = k;
	else
		m = options.m;
	end

	% Auxiliary shift to allow poles at zero:
	if min(abs(theta)) <= options.tol
		mu = + max(theta(theta~=inf)) + 1;
		mult = @(v) mult(v) - mu*v;
		theta = theta - mu;
	else
		mu = 0;
	end

	iter=0;
	s_conv = 1;

	errest = zeros(options.maxit, 2);		% error estimates
	check_idx = 0;							% number of convergence checks

	alpha = zeros(1, k);
	beta = zeros(1, k);
	y = zeros(n,1);
	zz = zeros(k,1);
	nrmb = norm(b,2);
	V = zeros(n, k+m+2);
	V(:, 1) = b/nrmb;
	invPoles = zeros(1, k);
	invPoles(1) = 0;
	T = [];
	Tcorr = 0;
	w = [];
	c=nrmb*eye(k+m,1);
	z_conv = [];
	maxnrmz = 0;

	[V(:,2),alpha(1),beta(1)] = short_recurrence_Arnoldi_in(mult,options.solveSystems{1},zeros(n,1),V(:,1),theta(1),inf,inf,0);
	iter=iter+1;
	if theta(1) == inf
		invPoles(2) = 0;
	else
		invPoles(2) = 1/theta(1);
	end

	[V(:,3),alpha(2),beta(2)] = short_recurrence_Arnoldi_in(mult,options.solveSystems{mod(1,length(theta))+1},...
		V(:,1),V(:,2),theta(mod(1,length(theta))+1),inf,theta(1),beta(1));
	iter=iter+1;
	if theta(mod(1,length(theta))+1) == inf
		check_idx = check_idx + 1;
		errest(check_idx, 1) = iter;
		[check_conv, T, Tcorr, w, z_conv, errest(check_idx, 2)] = check_convergence(alpha(s_conv:2), beta(s_conv:2),...
				invPoles(s_conv:2), T, Tcorr, w, f, c(1:2), z_conv, options.tol, maxnrmz, mu);
			s_conv = 3;
		if iter >= options.maxit         
			fprintf("maximum number of iterations (%d) reached \n", iter)
			z = V(:,1:2)*(z_conv);
			y = y + z;
			return
		end
		if check_conv
			z = V(:,1:2)*(z_conv);
			y = y + z;
			return
		end
		invPoles(3) = 0;
	else
		invPoles(3) = 1/theta(mod(1,length(theta))+1);
	end

	for i=3:k-1
		[V(:,i+1),alpha(i),beta(i)] = short_recurrence_Arnoldi_in(mult,options.solveSystems{mod(i-1,length(theta))+1},V(:,i-1),V(:,i),...
			theta(mod(i-1,length(theta))+1),theta(mod(i-3,length(theta))+1),theta(mod(i-2,length(theta))+1),...
			beta(i-1));
		iter=iter+1;
		if theta(mod(i-1,length(theta))+1) == inf
			check_idx = check_idx + 1;
			errest(check_idx, 1) = iter;	
			[check_conv, T, Tcorr, w, z_conv, errest(check_idx, 2)] = check_convergence(alpha(s_conv:i), beta(s_conv:i),...
				invPoles(s_conv:i), T, Tcorr, w, f, c(1 : i), z_conv, options.tol, maxnrmz, mu);
			s_conv = i+1;
			if iter >= options.maxit       
				fprintf("maximum number of iterations (%d) reached \n", iter)
				z = V(:,1:i)*(z_conv);
				y = y + z;
				return
			end
			if check_conv
				z = V(:,1:i)*(z_conv);
				y = y + z;
				return
			end
			invPoles(i+1) = 0;
		else
			invPoles(i+1) = 1/theta(mod(i-1,length(theta))+1);
		end
	end

	j=k-1;
	[V(:, k+1),alpha(k),beta(k)] = short_recurrence_Arnoldi_in(mult,mult,V(:,k-1),V(:,k),...
	inf,theta(mod(j-2,length(theta))+1),theta(mod(j-1,length(theta))+1),...
	beta(k-1));
	% Check if infinite pole is already present in theta:
	if (theta(mod(j,length(theta))+1) == inf)
		j = j+1;
	end

	iter=iter+1;
	check_idx = check_idx + 1;
	errest(check_idx, 1) = iter;	
	[check_conv, T, Tcorr, w, z_conv, errest(check_idx, 2)] = check_convergence(alpha(s_conv:k), beta(s_conv:k),...
				invPoles(s_conv:k), T, Tcorr, w, f, c(1 : k), z_conv, options.tol, maxnrmz, mu);
	if iter >= options.maxit    
		fprintf("maximum number of iterations (%d) reached \n", iter)
		z = V(:,1:k)*(z_conv);
		y = y + z;
		return
	end
	if check_conv
		z = V(:,1:k)*(z_conv);
		y = y + z;
		return
	end

	invPoles = zeros(1, m);
	invPoles(1) = 0;
	s_conv = 1;
	hatbeta = beta(k);
	alpha = zeros(1, m);
	beta = zeros(1, m);
	j=j+1;
	[V(:,k+2),alpha(1),beta(1)] = short_recurrence_Arnoldi_in(mult,options.solveSystems{mod(j-1,length(theta))+1},V(:,k),V(:,k+1),...
		theta(mod(j-1,length(theta))+1),theta(mod(j-2,length(theta))+1),inf,...
		hatbeta);
	iter=iter+1;
	if theta(mod(j-1,length(theta))+1) == inf
		check_idx = check_idx + 1;
		errest(check_idx, 1) = iter;	
		[check_conv, T, Tcorr, w, z_conv, errest(check_idx, 2)] = check_convergence(alpha(s_conv:1), beta(s_conv:1),...
				invPoles(s_conv:1), T, Tcorr, w, f, c(1 : k+1), z_conv, options.tol, maxnrmz, mu);
		s_conv = 2;
		if iter >= options.maxit     
			fprintf("maximum number of iterations (%d) reached \n", iter)
			z = V(:,1:k+1)*(z_conv);
			y = y + z;
			return
		end
		if check_conv
			z = V(:,1:k+1)*(z_conv);
			y = y + z;
			return
		end
		invPoles(2) = 0;
	else
		invPoles(2) = 1/theta(mod(j-1,length(theta))+1);
	end

	while j < options.maxit && ~check_conv
		j=j+1;
		[V(:,k+3),alpha(2),beta(2)] = short_recurrence_Arnoldi_in(mult,options.solveSystems{mod(j-1,length(theta))+1},V(:,k+1),V(:,k+2),...
			theta(mod(j-1,length(theta))+1),inf,theta(mod(j-2,length(theta))+1), beta(1));
		iter=iter+1;
		if theta(mod(j-1,length(theta))+1) == inf
			check_idx = check_idx + 1;
			errest(check_idx, 1) = iter;	
			[check_conv, T, Tcorr, w, z_conv, errest(check_idx, 2)] = check_convergence(alpha(s_conv:2), beta(s_conv:2),...
				invPoles(s_conv:2), T, Tcorr, w, f, c(1 : k+2), z_conv, options.tol, maxnrmz, mu);
			s_conv = 3;
			if iter >= options.maxit    
				fprintf("maximum number of iterations (%d) reached \n", iter)
				z = V(:,1:k+2)*(z_conv- [zz; zeros(size(z_conv,1)-size(zz,1),1)]);
				y = y + z;
				return
			end
			if check_conv
				z = V(:,1:k+2)*(z_conv- [zz; zeros(size(z_conv,1)-size(zz,1),1)]);
				y = y + z;
				return
			end
			invPoles(3) = 0;
		else
			invPoles(3) = 1/theta(mod(j-1,length(theta))+1);
		end

		for i=3:m-1
			j=j+1;
			[V(:,k+i+1),alpha(i),beta(i)] = short_recurrence_Arnoldi_in(mult,options.solveSystems{mod(j-1,length(theta))+1},V(:,k+i-1),V(:,k+i),...
				theta(mod(j-1,length(theta))+1),theta(mod(j-3,length(theta))+1),theta(mod(j-2,length(theta))+1),...
				beta(i-1));
			iter=iter+1;
			if theta(mod(j-1,length(theta))+1) == inf
				check_idx = check_idx + 1;
				errest(check_idx, 1) = iter;		
				[check_conv, T, Tcorr, w, z_conv, errest(check_idx, 2)] = check_convergence(alpha(s_conv:i), beta(s_conv:i),...
				invPoles(s_conv:i), T, Tcorr, w, f, c(1 : k+i), z_conv, options.tol, maxnrmz, mu);
			s_conv = i+1;
			if iter >= options.maxit    
				fprintf("maximum number of iterations (%d) reached \n", iter)
				z = V(:,1:k+i)*(z_conv- [zz; zeros(size(z_conv,1)-size(zz,1),1)]);
				y = y + z;
				return
			end   
			if check_conv
				z = V(:,1:k+i)*(z_conv- [zz; zeros(size(z_conv,1)-size(zz,1),1)]);
				y = y + z;
				return
			end 
				invPoles(i+1) = 0;
			else
				invPoles(i+1) = 1/theta(mod(j-1,length(theta))+1);
			end
		end
		[V(:,k+m+1),alpha(m),beta(m)] = short_recurrence_Arnoldi_in(mult,options.solveSystems{mod(j-2,length(theta))+1},V(:,k+m-1),V(:,k+m),...
			inf,theta(mod(j-2,length(theta))+1),theta(mod(j-1,length(theta))+1),...
			beta(m-1));
		% Check if infinite pole is already present in theta:
		if (theta(mod(j,length(theta))+1) == inf)
			j = j+1;
		end

		iter=iter+1;
		check_idx = check_idx + 1;
		errest(check_idx, 1) = iter;	
		[check_conv, T, Tcorr, w, z_conv, errest(check_idx, 2)] = check_convergence(alpha(s_conv:m), beta(s_conv:m),...
				invPoles(s_conv:m), T, Tcorr, w, f, c(1 : k+m), z_conv, options.tol, maxnrmz, mu);
		z1 = (z_conv- [zz; zeros(size(z_conv,1)-size(zz,1),1)]);
		z = V*[z1; zeros(2, 1)];
		y = y + z;
		if iter >= options.maxit     
			fprintf("maximum number of iterations (%d) reached \n", iter);
			return
		end 
		if check_conv
			return
		end 

		% ---------------- compression

		maxnrmz = max(maxnrmz, norm(z1));
		U = rational_krylov(T, w, xi, options);
		zz = (f(U'*T*U)*(U'*c));
		invPoles = zeros(1,m);
		invPoles(1) = 0;
		j=j+1;

		hatbeta = beta(m);
		alpha = zeros(1, m);
		beta = zeros(1, m);
		s_conv = 1;
		[V(:,k+m+2),alpha(1),beta(1)] = short_recurrence_Arnoldi_in(mult,options.solveSystems{mod(j-1,length(theta))+1},V(:,k+m),V(:,k+m+1),...
			theta(mod(j-1,length(theta))+1),theta(mod(j-2,length(theta))+1),inf,...
			hatbeta);
		iter=iter+1;
		V(:, 1:k+2) = V*blkdiag(U, eye(2));
		% V(:, k+3:k+m+2) = zeros(n, m);	% not necessary
		c(1:k) = U' * c;
		T = U'*T*U;
		w = U'*w;
		z_conv = zz;
		if theta(mod(j-1,length(theta))+1) == inf
			check_idx = check_idx + 1;
			errest(check_idx, 1) = iter;	
			[check_conv, T, Tcorr, w, z_conv, errest(check_idx, 2)] = check_convergence(alpha(s_conv:1), beta(s_conv:1),...
				invPoles(s_conv:1), T, Tcorr, w, f, c(1 : k+1), z_conv, options.tol, maxnrmz, mu);
			s_conv = 2;
			
			if iter >= options.maxit
			fprintf("maximum number of iterations (%d) reached \n", iter)
				z = V(:,1:k+1)*(z_conv- [zz; zeros(size(z_conv,1)-size(zz,1),1)]);
				y = y + z;
				return
			end  
			if check_conv 
				z = V(:,1:k+1)*(z_conv- [zz; zeros(size(z_conv,1)-size(zz,1),1)]);
				y = y + z;
				return
			end      
			invPoles(2) = 0;
		else
			invPoles(2) = 1/theta(mod(j-1,length(theta))+1);
		end
	end
end





