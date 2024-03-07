function [v3,alpha2,beta2] = short_recurrence_Arnoldi_in(mult, solveSystem, v1, v2, xi2, xi0, xi1, beta1, reorth)
	% Runs one iteration of rational Arnoldi for A Hermitian employing short recurrences
	%
	% Input:
	%	 mult			function handle such that mult(v) = A*v
	%	 solveSystem	function handle such that solveSystem(v) = (A - xi2*I)\v
	%			(only used if xi2 is not infinity)
	%	 v1, v2			last two basis vectors
	%	 xi2			pole for current iteration
	%	 xi0			pole for second to last iteration
	%	 xi1			pole for last iteration
	%	 beta1			current last entry of beta vector
	%	 reorth			additional columns for reorthogonalization (optional)
	%
	% Output:
	%	v3				next basis vector
	%	alpha2, beta2	next entries in alpha and beta vectors

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
	alpha2 = (r(:,1)'*v2)/(r(:,2)'*v2);
	w = r(:,1)-alpha2*r(:,2);
	if nargin >= 9
		w = w - reorth*(reorth'*w);
	end
	% % Reorthogonalization:
	% if nargin >= 9
	%     w = w - reorth*(reorth'*w);
	% end
	beta2 = norm(w,2);
	v3 = w / beta2;
end