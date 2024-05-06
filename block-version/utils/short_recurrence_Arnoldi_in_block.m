function [v3,alpha2,beta2] = short_recurrence_Arnoldi_in_block(mult, v1, v2, beta1, reorth)
	% Runs one iteration of block Lanczos A Hermitian
	%
	% Input:
	%	 mult			function handle such that mult(v) = A*v
	%	 solveSystem	function handle such that solveSystem(v) = (A - xi2*I)\v
	%			(only used if xi2 is not infinity)
	%	 v1, v2			last two basis block vectors
	%	 beta1			current last entry of beta vector
	%	 reorth			additional columns for reorthogonalization (optional)
	%
	% Output:
	%	v3				next basis vector
	%	alpha2, beta2	next entries in alpha and beta vectors

	tildey = mult(v2)-v1*beta1';
	alpha2 = tildey'*v2;
    alpha2 = (alpha2 + alpha2')/2;
	w = tildey-v2*alpha2;
	if nargin >= 5
		w = w - reorth*(reorth'*w);
	end
	% % Reorthogonalization:
	% if nargin >= 9
	%     w = w - reorth*(reorth'*w);
	% end
	[v3,beta2] = qr(w,0);
end