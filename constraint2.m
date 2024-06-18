function [c, ceq, gradc, gradceq] = constraint2(M0, Mf, positionsZ, bin_num, f, numK, a, M, noncom, grad)

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

if grad == "none"
% Construct propagator, propagate initial coeff vector
eM_tot = 1;
s = dt*squeeze(a(1:end-1, noncom(:, 1))) .* squeeze(a(2:end, noncom(:, 2)));
for q = 1:bin_num
    M_tot = sparse(0);
    for p=1:const_num
        M_tot = M_tot + M0(p).op*M0(p).ft(time_grid(q)+dt/2);
    end
    for p=1:ctrl_num
        M_tot = M_tot + Mf(p).op*Mf(p).ft(time_grid(q)+dt/2)*f(q,p);
    end
    eM_tot = eM_tot * expm(1j*dt*M_tot);
    at = tensorprod(a, eM_tot, 2, 2);
    ajk = squeeze(at(1:end-1, noncom(:, 1)));
    ajpl = squeeze(at(2:end, noncom(:, 2)));
    s = s + dt*ajk .* ajpl;
end
s = tensorprod(double(M), s, 2, 2);
c = sum(double(s).*conj(double(s)), 'all') - threshold;

elseif grad == "exact"
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
ajk = a(:, 1:end-1, noncom(:, 1));
ajpl = a(:, 2:end, noncom(:, 2));
s = sum(dt*ajk .* ajpl, 1);
s = tensorprod(double(M), s, 2, 3);
c = sum(double(s).*conj(double(s)), 'all') - threshold;

% Full gradient
ajk = a(:,1:end-1,noncom(:,1));
ajpl = a(:,2:end,noncom(:,2));
dajpl = da(:,:,:,2:end,noncom(:,2));
dajk = da(:,:,:,1:end-1,noncom(:,1));
s1 = sum(dt*conj(ajk) .* conj(ajpl), 1);
s1 = squeeze(tensorprod(s1, conj(double(M)), 3, 2));
s2 = sum(permute(repmat(dt*ajk,1,1,1,size(da,2),size(da,3)),[1,4,5,2,3]) .* dajpl + dt*dajk .* permute(repmat(ajpl,1,1,1,size(da,2),size(da,3)),[1,4,5,2,3]), 1);
s2 = squeeze(ttt(sptensor(s2), M, 5, 2));
s1 = permute(repmat(double(s1), 1, 1, size(s2,1), size(s2,2)),[3,4,1,2]);
gradc = 2*sum(real(double(s1 .* s2)), [3,4]);
gradc = [gradc(:); 0];

elseif grad == "finite"
% Construct propagator, propagate initial coeff vector
eM_tot = 1;
s = dt*squeeze(a(1:end-1, noncom(:, 1))) .* squeeze(a(2:end, noncom(:, 2)));
for q = 1:bin_num
    M_tot = sparse(0);
    for p=1:const_num
        M_tot = M_tot + M0(p).op*M0(p).ft(time_grid(q)+dt/2);
    end
    for p=1:ctrl_num
        M_tot = M_tot + Mf(p).op*Mf(p).ft(time_grid(q)+dt/2)*f(q,p);
    end
    eM_tot = eM_tot * expm(1j*dt*M_tot);
    at = tensorprod(a, eM_tot, 2, 2);
    ajk = squeeze(at(1:end-1, noncom(:, 1)));
    ajpl = squeeze(at(2:end, noncom(:, 2)));
    s = s + dt*ajk .* ajpl;
end
s = tensorprod(double(M), s, 2, 2);
c = sum(double(s).*conj(double(s)), 'all') - threshold;

dx = 1e-10;
gradc = zeros(bin_num,ctrl_num);
for qq = 1:bin_num
    for pp = 1:ctrl_num
        ff = f; ff(qq,pp) = f(qq,pp) + dx;
        eM_tot = 1;
        sf = dt*squeeze(a(1:end-1, noncom(:, 1))) .* squeeze(a(2:end, noncom(:, 2)));
        for q = 1:bin_num
            M_tot = sparse(0);
            for p=1:const_num
                M_tot = M_tot + M0(p).op*M0(p).ft(time_grid(q)+dt/2);
            end
            for p=1:ctrl_num
                M_tot = M_tot + Mf(p).op*Mf(p).ft(time_grid(q)+dt/2)*ff(q,p);
            end
            eM_tot = eM_tot * expm(1j*dt*M_tot);
            at = tensorprod(a, eM_tot, 2, 2);
            ajk = squeeze(at(1:end-1, noncom(:, 1)));
            ajpl = squeeze(at(2:end, noncom(:, 2)));
            sf = sf + dt*ajk .* ajpl;
        end
        sf = tensorprod(double(M), sf, 2, 2);
        cf = sum(double(sf).*conj(double(sf)), 'all') - threshold;
        gradc(qq,pp) = (cf - c)/dx;
    end
end
gradc = [gradc(:);0];

end
end