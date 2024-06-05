function [c, ceq, gradc, gradceq] = constraint_T(M0, Mf, positionsZ, bin_num, f, numK, noncom, grad)

ceq = [];
gradceq = [];

if grad == false

    T=f(end);
    f=f(1:end-1);
    time_grid=linspace(0,T,bin_num+1);
    c = constraint(M0, Mf, positionsZ, time_grid, f, numK, noncom, false);
    gradc = [];

else

    T=f(end);
    f=f(1:end-1);
    time_grid=linspace(0,T,bin_num+1);
    dt=10^(-10);
    [c, ceq, gradc, gradceq] = constraint(M0, Mf, positionsZ, time_grid, f, numK, noncom, true);
    time_grid2 = linspace(0,T+dt,bin_num+1);
    c2 = constraint(M0, Mf, positionsZ, time_grid2, f, numK, noncom, false);
    iGT = (c2 - c)/dt;
    gradc = [gradc; iGT];

end
end