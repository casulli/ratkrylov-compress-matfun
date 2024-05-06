function U = rational_krylov_block(A, W, poles, options)
	% constructs rational Krylov subspace Q(A, W, poles)
	% by default the starting vector W is not included (i.e., no starting infinite pole is added) 
	
    bs = size(W,2);
	if nargin < 4
		options = [];
	end

	if isfield (options, 'isreal') == 0
		options.isreal = 0;
	end

	if abs(poles(1)) == inf
		[W,~] = qr(W,0);
		j = 2;
		U = W;
	elseif options.isreal && poles(2) == conj(poles(1))
	    [W,~] = qr(W,0);
		W = (A - poles(1)*speye(size(A))) \  W;
		W1=real(W);
		W=imag(W);
		j = 3;
		[U,~] = qr([W1, W], 0);
        W=U(:,end-bs+1:end);
	else
		[W,~] = qr(W,0);
		W = (A - poles(1)*speye(size(A))) \  W;
		j = 2;
		[W,~] = qr(W,0);
		U = W;
	end

	while j <= length(poles)
		if abs(poles(j)) == inf
			W = A * W;
			j = j+1;
		elseif options.isreal && j+1<=length(poles) && poles(j+1) == conj(poles(j))
			W = (A - poles(j)*speye(size(A))) \  W;
			W1 = real(W);
			W = imag(W);
			
			% cgs2
			W1 = W1 - U * (U'*W1);
			W1 = W1 - U * (U'*W1);
			[W1,~] = qr(W1,0);
			U = [U, W1];

			% qr alternative
			% [U,~] = qr([U, W1], 0);
			j = j+2;
		else
			W = (A - poles(j)*speye(size(A))) \  W;
			j = j+1;
		end

		% cgs2
		W = W - U * (U'*W);
		W = W - U * (U'*W);
        [W,~] = qr(W,0);
		U = [U, W];

		% qr alternative
		% [U,~] = qr([U, W], 0);

        W = U(:,end-bs+1:end);
	end
end