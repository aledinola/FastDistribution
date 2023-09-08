function [StationaryDistKron] = StationaryDist_Case1_Iteration_raw_AD(StationaryDistKron,PolicyKron,n_d,n_a,n_z,pi_z,simoptions)

StationaryDistKron = gather(StationaryDistKron);
PolicyKron         = gather(PolicyKron);

StationaryDist = reshape(StationaryDistKron,[n_a,n_z]);
Policy         = reshape(PolicyKron,[n_a,n_z]);

tolerance = simoptions.tolerance;
maxit     = simoptions.maxit;
verbose   = simoptions.verbose;

iter = 1;
err  = tolerance+1;

while err>tolerance && iter<=maxit


    %% Step 1: Update endogenous state from (a,z) to a'
    % mu(a,z) ==> mu_hat(a',z)

    mu_hat = zeros(n_a,n_z);
    for z_c=1:n_z
        for a_c=1:n_a
            mu_hat(Policy(a_c,z_c),z_c)=mu_hat(Policy(a_c,z_c),z_c)+StationaryDist(a_c,z_c);
        end
    end


    %% Step 2: update exogenous state from z to z', using exog transition matrix pi_z
    % mu_hat(a',z)*pi_z(z,z')  => mu_new(a',z')

    StationaryDistNew = mu_hat*pi_z;

    err = max(abs(StationaryDistNew(:)-StationaryDist(:)));

    if verbose==1
        fprintf('iter = %d, err = %f \n',iter,err)
    end

    % Update
    StationaryDist = StationaryDistNew;
    iter = iter+1;

end %end while loop

end %end function