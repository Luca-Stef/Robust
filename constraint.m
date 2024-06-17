function [c, ceq, gradc, gradceq] = constraint(M0, Mf, positionsZ, bin_num, f, numK, noncom, grad)

threshold = 1e-2;
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

% Auxiliary matrix
rows = [];
sizes = cellfun(@(x) size(x, 1), noncom);
noncom2 = [];
for i = 1:size(noncom,1)
    rows = [rows; i*ones(sizes(i),1)];
    noncom2 = [noncom2; noncom{i}];
end
M = sptensor(sparse(rows, 1:sum(sizes), ones(sum(sizes),1)));

if grad == false
% Construct propagator, propagate initial coeff vector
eM_tot = 1;
phases = diag(exp(1j*noncom2(:, 3)));
s = (dt*squeeze(a(1, 1:end-1, noncom2(:, 1))) * phases) .* squeeze(a(1, 2:end, noncom2(:, 2)));
for q = 1:bin_num
    M_tot = sparse(0);
    for p=1:const_num
        M_tot = M_tot + M0(p).op*M0(p).ft(time_grid(q)+dt/2);
    end
    for p=1:ctrl_num
        M_tot = M_tot + Mf(p).op*Mf(p).ft(time_grid(q)+dt/2)*f(q,p);
    end
    eM_tot = eM_tot * expm(1j*dt*M_tot);
    a(q+1,:,:) = tensorprod(squeeze(a(q+1,:,:)), eM_tot, 2, 2);
    ajk = squeeze(a(q+1, 1:end-1, noncom2(:, 1)));
    ajpl = squeeze(a(q+1, 2:end, noncom2(:, 2)));
    s = s + (dt*ajk * phases) .* ajpl;
end
s = ttt(M, sptensor(s), 2, 2);
c = sum(double(s).*conj(double(s)), 'all') - threshold;

else
% Compute and propagate derivatives
% da derivative of coefficient vector of perturbation j wrt to
% driving of control p in time bin q
% size: [time t, driving at time interval q, control operator p, perturbation j, algebra dim]
D = zeros(bin_num, ctrl_num, gen_num, gen_num);
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
    for i = q+1:bin_num+1
        a(i,:,:) = tensorprod(a(i,:,:), expm(1j*dt*M_tot), 3,2);
        da(i,q+1:i-1,:,:,:) = tensorprod(da(i,q+1:i-1,:,:,:), expm(1j*dt*M_tot), 5,2);
        da(i,q,:,:,:) = permute(squeeze(tensorprod(D(q,:,:,:), a(i,:,:), 4, 3)), [1,3,2]);
    end
end

% Compute cost
ajk = a(:, 1:end-1, noncom2(:, 1));
ajpl = a(:, 2:end, noncom2(:, 2));
phases = diag(exp(1j*noncom2(:, 3)));
s = sum(tensorprod(dt*ajk, phases, 3, 1) .* ajpl, 1);
s = ttt(M, sptensor(s), 2, 3);
c = sum(double(s).*conj(double(s)), 'all') - threshold;

% Full gradient
phases = diag(exp(1j*noncom2(:,3)));
ajk = a(:,1:end-1,noncom2(:,1));
ajpl = a(:,2:end,noncom2(:,2));
dajpl = da(:,:,:,2:end,noncom2(:,2));
dajk = da(:,:,:,1:end-1,noncom2(:,1));
s1 = sum(tensorprod(dt*conj(ajk), conj(phases),3,1) .* conj(ajpl), 1);
s1 = squeeze(ttt(sptensor(s1), M, 3, 2));
s2 = sum(permute(repmat(tensorprod(dt*ajk, phases,3,1),1,1,1,size(da,2),size(da,3)),[1,4,5,2,3]) .* dajpl + tensorprod(dt*dajk, phases, 5, 1) .* permute(repmat(ajpl,1,1,1,size(da,2),size(da,3)),[1,4,5,2,3]), 1);
s2 = squeeze(ttt(sptensor(s2), M, 5, 2));
s1 = permute(repmat(double(s1), 1, 1, size(s2,1), size(s2,2)),[3,4,1,2]);
gradc = 2*sum(real(double(s1 .* s2)), [3,4]);
gradc = [gradc(:); 0];

end
end