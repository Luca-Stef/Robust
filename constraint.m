function [c, ceq, gradc, gradceq] = constraint(M0, Mf, positionsZ, bin_num, f, numK, noncom, grad)

threshold = 1e-3; %0
ceq = [];
gradceq = [];
gradc = [];
T=f(end);
f=f(1:end-1);
time_grid = linspace(0,T,bin_num+1);
dt = time_grid(2) - time_grid(1);
n = length(positionsZ);
const_num=length(M0);
ctrl_num = n+1;
gen_num = 2*n^2+3*n+1;
f = reshape(f(1:end), [bin_num, ctrl_num]);
a = zeros(bin_num+1, n, gen_num); % initial coefficient vectors of perturbations. size: [bin_num, # of qubits, algebra dimension]
for j = 1:n
    a(:,j,positionsZ(j)) = ones(size(a,1),1);
end

if grad == false
% Construct propagator, propagate initial coeff vector
for q = bin_num:-1:1
    M_tot = sparse(0);
    for p=1:const_num
        M_tot = M_tot + M0(p).op*M0(p).ft(time_grid(q)+dt/2);
    end
    for p=1:ctrl_num
        M_tot = M_tot + Mf(p).op*Mf(p).ft(time_grid(q)+dt/2)*f(q,p);
    end
    for j = 1:n %% n-1
        parfor i = q+1:bin_num+1
            a(i,j,:) = expvcpu(dt,-M_tot,squeeze(a(i,j,:)),numK);
        end
    end
end

% Compute cost
c = -threshold;
parfor group = 1:size(noncom,1)
    %for j = 1:n-1 
        %s = 0;
        %for kl = 1:size(noncom{group},1)
        %    k = noncom{group}(kl,1);
        %    l = noncom{group}(kl,2);
        %    phase = exp(1j*noncom{group}(kl,3));
        %    s = s + phase*sum(dt*a(:,j,k).*a(:,j+1,l));
        %end
        %s = sum((squeeze(dt*a(:,j,noncom{group}(:,1))) * diag(exp(1j*noncom{group}(:,3)))) .* squeeze(a(:,j+1,noncom{group}(:,2))),'all');
        %c = c + s*conj(s);
    %end
    s = sum(tensorprod(squeeze(dt*a(:, 1:end-1, noncom{group}(:, 1))), diag(exp(1j*noncom{group}(:, 3))), 3, 1) .* squeeze(a(:, 2:end, noncom{group}(:, 2))), [1, 3]);
    c = c + sum(s.*conj(s));
end

else
D = zeros(bin_num, ctrl_num, gen_num, gen_num);
% Compute and propagate derivatives
% da derivative of coefficient vector of perturbation j wrt to
% driving of control p in time bin q
% size: [time t, driving at time interval q, control operator p, perturbation j, algebra dim]
da = zeros(bin_num+1, bin_num, ctrl_num, n, gen_num);
for q = bin_num:-1:1
        % Construct propagator
        M_tot = sparse(0);
    for p=1:const_num
        M_tot = M_tot + M0(p).op*M0(p).ft(time_grid(q)+dt/2);
    end
    for p=1:ctrl_num
        M_tot = M_tot + Mf(p).op*Mf(p).ft(time_grid(q)+dt/2)*f(q,p);
    end
    tq = time_grid(q);  
    parfor p = 1:ctrl_num
        C1 = Mf(p).op * Mf(p).ft(tq+dt/2);
        C2 = M_tot * C1 - C1 * M_tot;
        D(q,p,:,:) = 1i * dt * C1 - (dt^2/2) * C2; 
    end
    for j = 1:n %% n-1
        for i = q+1:bin_num+1
            % propagate initial coeff vector
            a(i,j,:) = expvcpu(dt,-M_tot,squeeze(a(i,j,:)),numK);
            for p = 1:ctrl_num
                da(i,q,p,j,:) = squeeze(D(q,p,:,:))*squeeze(a(i,j,:));
                parfor ii = q+1:i-1
                    da(i,ii,p,j,:) = expvcpu(dt,-M_tot,squeeze(da(i,ii,p,j,:)),numK);
                end
            end
        end
    end
end

% Compute cost
c = -threshold;
parfor group = 1:size(noncom,1)
    %for j = 1:n-1 
        %s = 0;
        %for kl = 1:size(noncom{group},1)
        %    k = noncom{group}(kl,1);
        %    l = noncom{group}(kl,2);
        %    phase = exp(1j*noncom{group}(kl,3));
        %    s = s + phase*sum(dt*a(:,j,k).*a(:,j+1,l));
        %end
        %s = sum((squeeze(dt*a(:,j,noncom{group}(:,1))) * diag(exp(1j*noncom{group}(:,3)))) .* squeeze(a(:,j+1,noncom{group}(:,2))),'all');
        %c = c + s*conj(s);
    %end
    s = sum(tensorprod(squeeze(dt*a(:, 1:end-1, noncom{group}(:, 1))), diag(exp(1j*noncom{group}(:, 3))), 3, 1) .* squeeze(a(:, 2:end, noncom{group}(:, 2))), [1, 3]);
    c = c + sum(s.*conj(s));
end

% Full gradient
gradc = zeros(bin_num,ctrl_num);    
for group = 1:size(noncom,1)    
    %for q = 1:bin_num
        %for p = 1:ctrl_num
            %for j = 1:n-1 
                %s1 = 0;
                %s2 = 0;
                %for kl = 1:size(noncom{group},1)
                %    k = noncom{group}(kl,1);
                %    l = noncom{group}(kl,2);
                %    phase = exp(1j*noncom{group}(kl,3));
                %    s1 = s1 + conj(phase)*sum(dt*conj(a(:,j,k).*a(:,j+1,l)));
                %    s2 = s2 + phase*sum(dt*a(:,j,k).*da(:,q,p,j+1,l) + dt*da(:,q,p,j,k).*a(:,j+1,l));
                %end
            %end
            phases = diag(exp(1j*noncom{group}(:,3)));
            ajk = squeeze(a(:,1:end-1,noncom{group}(:,1)));
            ajpl = squeeze(a(:,2:end,noncom{group}(:,2)));
            dajpl = squeeze(da(:,:,:,2:end,noncom{group}(:,2)));
            dajk = squeeze(da(:,:,:,1:end-1,noncom{group}(:,1))); 
            s1 = sum(tensorprod(dt*conj(ajk), conj(phases),3,1) .* conj(ajpl), [1,3]);                                                                    
            s2 = squeeze(sum(permute(repmat(tensorprod(dt*ajk, phases,3,1),1,1,1,size(da,2),size(da,3)),[1,4,5,2,3]) .* dajpl + tensorprod(dt*dajk, phases, 5, 1) .* permute(repmat(ajpl,1,1,1,size(da,2),size(da,3)),[1,4,5,2,3]), [1,5]));
            gradc = gradc + 2*real(squeeze(tensorprod(s1,s2,2,3))); 
        %end
    %end
end
    
gradc = gradc(:);
gradc = [gradc; 0];

end
end