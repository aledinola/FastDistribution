function [StationaryDistKron] = StationaryDist_Case1_Iteration_raw_AD_vec(StationaryDistKron,PolicyKron,n_d,n_a,n_z,pi_z,simoptions)

StationaryDistKron = gather(StationaryDistKron);
PolicyKron         = gather(PolicyKron);

StationaryDist = reshape(StationaryDistKron,[n_a,n_z]);
Policy         = reshape(PolicyKron,[n_a,n_z]);

tolerance = simoptions.tolerance;
maxit     = simoptions.maxit;
verbose   = simoptions.verbose;

%% Step 1: Build transition operator from (a,z) to (a',z)
% Note: we update only endogenous state a but NOT the exog state z

n_all = n_a*n_z;
Tmat = sparse(n_all,n_all);

xx = (1:n_a)';

% Tmat is (a,z)=>(a',z)
for z_c = 1:n_z
    ii = (z_c-1)*n_a+1:z_c*n_a;
    Tmat(ii,ii) = sparse(xx,Policy(:,z_c),ones(n_a,1),n_a,n_a);
end

Tmat_transpose = Tmat'; % 

StationaryDistVec = StationaryDist(:);

iter = 1;
err  = tolerance+1;

while err>tolerance && iter<=maxit

    StationaryDistMid = Tmat_transpose*StationaryDistVec; %(n_a*n_z,1)
    StationaryDistVec_new = vec(reshape(StationaryDistMid,[n_a,n_z])*pi_z);
    %StationaryDistMid2 = reshape(StationaryDistMid,[n_a,n_z]);
    %StationaryDistFinal = StationaryDistMid2*pi_z; %(n_a,n_z)
    %StationaryDistVec_new = StationaryDistFinal(:);

    err = max(abs(StationaryDistVec_new-StationaryDistVec));

    if verbose==1
        fprintf('iter = %d, err = %f \n',iter,err)
    end

    % Update
    StationaryDistVec = StationaryDistVec_new;
    iter = iter+1;

end %end while loop

end %end function

%%%% Small subfunction to vectorize array
function x_vec = vec(x_array)

x_vec = x_array(:);

end