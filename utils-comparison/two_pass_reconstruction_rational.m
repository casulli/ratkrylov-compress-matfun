function x = two_pass_reconstruction_rational(mult, solveSystems, b, alpha, beta, xi, y)
	% Reconstructs solution for two-pass Lanczos efficiently in the second pass

	n = length(b);
	v1 = zeros(n, 1);
	v2 = b/norm(b);
	xi0 = inf;
	xi1 = inf;
	xi2 = xi(1);
	beta1 = 0;
	solveSystem = solveSystems{1};
	if xi0 == inf
		tildey = mult(v2)-beta1*v1;
	else
		tildey = mult(v2+(beta1/xi0)*v1)-beta1*v1;
	end
	if xi1 == xi2
		if xi2 == inf
			r(:,1) = tildey;
		else
			r(:,1) = -solveSystem(tildey)*xi2;
		end
		r(:,2) = v2;
	elseif xi1 == inf
		% xi2 is finite here
		r = -solveSystem([tildey,v2])*xi2;
	else
		if xi2 == inf
			r = [tildey,v2-mult(v2/xi1)];
		else
			r = -solveSystem([tildey,v2-mult(v2/xi1)])*xi2;
		end
	end
	w = r(:,1) - alpha(1)*r(:,2);
	v = w / beta(1);
	x = y(1)*v2 + y(2)*v;

	v1 = v2;
	v2 = v;
	xi0 = xi1;
	xi1 = xi2;
	xi2 = xi(2);
	beta1 = beta(1);
	solveSystem = solveSystems{2};
	for i = 3:length(y)
		if xi0 == inf
			tildey = mult(v2)-beta1*v1;
		else
			tildey = mult(v2+(beta1/xi0)*v1)-beta1*v1;
		end
		
		if xi1 == xi2
			if xi2 == inf
				r(:,1) = tildey;
			else
				r(:,1) = -solveSystem(tildey)*xi2;
			end
			r(:,2) = v2;
		elseif xi1 == inf
			% xi2 is finite here
			r = -solveSystem([tildey,v2])*xi2;
		else
			if xi2 == inf
				r = [tildey,v2-mult(v2/xi1)];
			else
				r = -solveSystem([tildey,v2-mult(v2/xi1)])*xi2;
			end
		end
		
		w = r(:,1) - alpha(i-1)*r(:,2);
		v = w / beta(i-1);

		x = x + y(i)*v;

		v1 = v2;
		v2 = v;
		xi0 = xi1;
		xi1 = xi2;
		xi2 = xi(i);
		beta1 = beta(i-1);
		solveSystem = solveSystems{i};	
	end
	

end