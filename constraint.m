function [c, ceq, gradc, gradceq] = constraint(M0, Mf, positionsZ, time_grid, f, numK, noncom, grad)

threshold = 1e-3; %0
ceq = [];
gradceq = [];
gradc = [];
bin_num=length(time_grid)-1;
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

%a = zeros(bin_num+1, n-1, gen_num); % initial coefficient vectors of perturbations. size: [bin_num, # of qubits, algebra dimension]
%for j = 1:n-1
%    a(1,j,positionsXX(j)) = 1;    % normalise???
%end

if grad == false
% Construct propagator, propagate initial coeff vector
M_tot = {};
for q = bin_num:-1:1
    M_tot{q} = sparse(0);
    for p=1:const_num
        M_tot{q} = M_tot{q} + M0(p).op*M0(p).ft(time_grid(q)+dt/2);
    end
    for p=1:ctrl_num
        M_tot{q} = M_tot{q} + Mf(p).op*Mf(p).ft(time_grid(q)+dt/2)*f(q,p);
    end
    %for j = 1:n %% n-1
    %    a(q+1,j,:) = expvcpu(dt,M_tot,reshape(a(q,j,:),[],1),numK);
    %end
    for j = 1:n %% n-1
        parfor i = q+1:bin_num+1
            a(i,j,:) = expvcpu(dt,-M_tot{q},squeeze(a(i,j,:)),numK);
        end
    end
end

% Compute cost
c = -threshold;
for j = 1:n-1 % n-2
    parfor group = 1:size(noncom,1)
        s = 0;
        for kl = 1:size(noncom{group},1)
            k = noncom{group}(kl,1);
            l = noncom{group}(kl,2);
            phase = exp(1j*noncom{group}(kl,3));
            s = s + phase*sum(dt*a(:,j,k).*a(:,j+1,l));
        end
        c = c + s*conj(s);
    end
end

else
M_tot = {};
D = zeros(bin_num, ctrl_num, gen_num, gen_num);
% Compute and propagate derivatives
% da derivative of coefficient vector of perturbation j wrt to
% driving of control p in time bin q
% size: [time t, driving at time interval q, control operator p, perturbation j, algebra dim]
da = zeros(bin_num+1, bin_num, ctrl_num, n, gen_num);
for q = bin_num:-1:1
        % Construct propagator
        M_tot{q} = sparse(0);
    for p=1:const_num
        M_tot{q} = M_tot{q} + M0(p).op*M0(p).ft(time_grid(q)+dt/2);
    end
    for p=1:ctrl_num
        M_tot{q} = M_tot{q} + Mf(p).op*Mf(p).ft(time_grid(q)+dt/2)*f(q,p);
    end
    tq = time_grid(q);  
    parfor p = 1:ctrl_num
        C1 = Mf(p).op * Mf(p).ft(tq+dt/2);
        C2 = M_tot{q} * C1 - C1 * M_tot{q};
        D(q,p,:,:) = 1i * dt * C1 - (dt^2/2) * C2;
    end  
    for j = 1:n %% n-1
        for i = q+1:bin_num+1
            % propagate initial coeff vector
            a(i,j,:) = expvcpu(dt,-M_tot{q},squeeze(a(i,j,:)),numK);
            for p = 1:ctrl_num
                da(i,q,p,j,:) = squeeze(D(q,p,:,:))*squeeze(a(i,j,:));
                parfor ii = q+1:i-1
                    da(i,ii,p,j,:) = expvcpu(dt,-M_tot{q},squeeze(da(i,ii,p,j,:)),numK);
                end
            end
        end
    end
end

% Compute cost
c = -threshold;
for j = 1:n-1 % n-2
    parfor group = 1:size(noncom,1)
        s = 0;
        for kl = 1:size(noncom{group},1)
            k = noncom{group}(kl,1);
            l = noncom{group}(kl,2);
            phase = exp(1j*noncom{group}(kl,3));
            s = s + phase*sum(dt*a(:,j,k).*a(:,j+1,l));
        end
        c = c + s*conj(s);
    end
end

% Full gradient
gradc = zeros(bin_num,ctrl_num);    
for q = 1:bin_num
    for p = 1:ctrl_num
        for j = 1:n-1 % n-2
            for group = 1:size(noncom,1)
                s1 = 0;
                s2 = 0;
                for kl = 1:size(noncom{group},1)
                    k = noncom{group}(kl,1);
                    l = noncom{group}(kl,2);
                    phase = exp(1j*noncom{group}(kl,3));
                    s1 = s1 + conj(phase)*sum(dt*conj(a(:,j,k).*a(:,j+1,l)));
                    s2 = s2 + phase*sum(dt*a(:,j,k).*da(:,q,p,j+1,l) + dt*da(:,q,p,j,k).*a(:,j+1,l));
                end
                gradc(q,p) = gradc(q,p) + 2*real(s1*s2);
            end
        end
    end
end
    
gradc = gradc(:);

end
end