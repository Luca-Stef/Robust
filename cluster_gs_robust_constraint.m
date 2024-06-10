function cluster_gs_robust_constraint(dg, start, step, stop, grad)
addpath('functions/');
maxfun=10^4; %maximum number of function evaluations in the optimisation defualt 4
nw=4;%number of workers
iF_target=10^(-3); %target infidelity, optimisation halted if infidelity drops below this default -5
parpool('local',nw);%starting parallel local pool
maxNumCompThreads(nw);
islanczos=1;
numK=25;
n_list = start:step:stop; %list of qubit numbers
F_list=zeros(1,length(n_list));% list of fidelity
f_list=cell(1,length(n_list));% list of control amplitudes
t_list=cell(1,length(n_list));% list of time grid for time evolution
for jn=1:length(n_list)
n=n_list(jn);
gen_num=2*n^2+3*n+1;%dimension of the operator space
qubit_num=n; %qubit number
J=ones(1,n-1); %interaction strengh, m by (n-1) array, n: number of qubits, m: number of sampling points in the hypercube
T0=n*pi/2; % initial guess for the duration
bin_num=10*n; % number of time bins in the control pulse
Ntry=4;%number of initial guesses
fprintf('qubit number:%d\n',n);
%load structure constants 
genlist=load(['structure_constant/obcN',num2str(qubit_num),'genlist']);
positionsZ=genlist(1:n);
positionsXX=genlist(n+1:2*n-1);
positionsX1=genlist(2*n);
positionsXn=genlist(2*n+1);
positionsZX=genlist(2*n+2);
positionsXZX=genlist(2*n+3:3*n);
positionsXZ=genlist(3*n+1);
positionsYZZZY=genlist(3*n+2);
positionsZZZ=genlist(3*n+3);
Kxx=cell(1,n-1);
Kxxsum=sparse(gen_num,gen_num);
for jj=1:n-1
K=load(['structure_constant/obcN',num2str(qubit_num),'Kxx',num2str(jj)]);
Kxx{jj}=1i*sparse(K(:,1),K(:,2),K(:,3),gen_num,gen_num);
Kxxsum=Kxxsum+Kxx{jj};
if ~ishermitian(Kxx{jj})
 fprintf('warning: Kxx not hermitian\n');
end
end
Kz=cell(1,n);
Kzsum=sparse(gen_num,gen_num);
for jj=1:n
K=load(['structure_constant/obcN',num2str(qubit_num),'Kz',num2str(jj)]);
Kz{jj}=1i*sparse(K(:,1),K(:,2),K(:,3),gen_num,gen_num);
Kzsum=Kzsum+Kz{jj};
if ~ishermitian(Kz{jj})
 fprintf('warning: Kz not hermitian\n');
end
end
Kx=cell(1,n);
Kxsum=sparse(gen_num,gen_num);
for jj=[1,n]
K=load(['structure_constant/obcN',num2str(qubit_num),'Kx',num2str(jj)]);
Kx{jj}=1i*sparse(K(:,1),K(:,2),K(:,3),gen_num,gen_num);
Kxsum=Kxsum+Kx{jj};
if ~ishermitian(Kx{jj})
 fprintf('warning: Kx not hermitian\n');
end
end
fprintf('optimisation\n');
const_num=1;
ctrl_num=n+1;
M0=struct('op',cell(const_num,1),'ft',cell(const_num,1)); %three single qubit terms and 2 interaction term
Mf=struct('op',cell(ctrl_num,1),'ft',cell(ctrl_num,1));
count=1;
M0(count).op=Kxxsum;
M0(count).ft=@(x) 1;
for count=1:n
Mf(count).op=Kz{count};
Mf(count).ft=@(x) 1;
end
Mf(n+1).op=Kxsum; 
Mf(n+1).ft=@(x) 1;
c0=zeros(gen_num,1);%initial coefficient vector
c0(positionsZ)=1;
c0=c0/norm(c0);
ctg=zeros(gen_num,1);
ctg(positionsZX)=1;%target coefficient vector
ctg(positionsXZX)=1;
ctg(positionsXZ)=1;
ctg=ctg/norm(ctg);
A=[];
b=[];
Aeq=[];
beq=[];
nonlcon=[];
fmax=Inf;
f0=rand(bin_num,ctrl_num);
f0=f0(:);
x0=[f0;T0];
lb=-fmax*ones(length(x0),1);
ub=fmax*ones(length(x0),1);
fun = @(f) infid_grape_lanczos_T_robust(J,Kxx,Mf,c0,ctg,bin_num,f,numK);
noncom = load(sprintf("noncom%d.mat", n)); noncom = noncom.Expression1; % list of operators that don't commute with initial condition
nonlcon = @(f) constraint(M0, Mf, positionsZ, bin_num, f, numK, noncom, grad);
options = optimoptions('fmincon','SpecifyObjectiveGradient',true,'SpecifyConstraintGradient',grad,'Display','iter');
options.MaxIterations = 500;
options.MaxFunctionEvaluations=500;
options.ObjectiveLimit=iF_target;

startpath = '';
if startpath == ""
if jn==1
    xL=cell(Ntry,1);
    FL=zeros(Ntry,1);
    for ii=1:Ntry
        f0=2*(rand(bin_num,ctrl_num)-1/2)*0.1;
        f0=f0(:);
        x0=[f0;T0];
        fprintf("Pre-optimisation %d",ii)
        x1=fmincon(fun,x0,A,b,Aeq,beq,lb,ub,nonlcon,options);
        xL{ii}=x1;
        FL(ii)=1-fun(x1);
    end
    [Fmax,imax]=max(FL);
    x1=xL{imax};
else
    bin_num_prev=length(t_list{jn-1})-1;
    ctrl_num_prev=n_list(jn-1)+1;
    f_prev=reshape(f_list{jn-1},[bin_num_prev,ctrl_num_prev]);
    T_prev=t_list{jn-1}(end);
    fprintf('previous duration: %d\n',T_prev/(pi/2));
    f0=2*(rand(bin_num,ctrl_num)-1/2)*0.1;
    f0(1:bin_num_prev,1:ctrl_num_prev)=f_prev;
    f0=f0(:);
    x0=[f0;T0];
    fprintf("Pre-optimisation")
    x1=fmincon(fun,x0,A,b,Aeq,beq,lb,ub,nonlcon,options);
end

else
    fprintf("Resuming with solution from %s\n", startpath);
    prev = load(startpath, "f_list", "t_list", "n_list");
    bin_num_prev=length(prev.t_list{1})-1;
    ctrl_num_prev = prev.n_list + 1;
    f_prev=reshape(prev.f_list{1},[bin_num_prev,ctrl_num_prev]);
    f0=2*(rand(bin_num,ctrl_num)-1/2)*0.1;
    f0(1:bin_num_prev,1:ctrl_num_prev)=f_prev;
    f0=f0(:);
    x1=[f0;T0];
end

%refine with more iterations
fprintf("Refinement")
options = optimoptions('fmincon','SpecifyObjectiveGradient',true,'SpecifyConstraintGradient',grad,'Display','iter');
options.MaxIterations = maxfun;
options.MaxFunctionEvaluations=maxfun;
options.ObjectiveLimit=iF_target;
x_optm=fmincon(fun,x1,A,b,Aeq,beq,lb,ub,nonlcon,options);
F_list(jn)=1-fun(x_optm);
f_list{jn}=x_optm(1:end-1);
t_list{jn}=linspace(0,x_optm(end),bin_num+1);
iF_cluster_log=log10(1-F_list(jn));

% Validation with n_sample random hypercube points
fprintf("Validation %d qubits %g error", n, dg);
%sample_infidelity = infid_grape_lanczos_T_robust(Jerr,Kxx,Mf,c0,ctg,bin_num,x_rob,numK)
Jerr = J_error("centers", n-1, dg, 100);
%centers_infidelity = infid_grape_lanczos_T_robust(Jerr,Kxx,Mf,c0,ctg,bin_num,x_rob,numK)

Jrand = J_error("random", n-1, dg, 1000);
%rand_infidelity1000 = infid_grape_lanczos_T_robust(Jrand,Kxx,Mf,c0,ctg,bin_num,x_rob,numK)
%non_robust_rand_inf1000 = infid_grape_lanczos_T_robust(Jrand,Kxx,Mf,c0,ctg,bin_num,x_optm,numK)
Jrand = J_error("random", n-1, dg, 100);
%rand_infidelity100 = infid_grape_lanczos_T_robust(Jrand,Kxx,Mf,c0,ctg,bin_num,x_rob,numK)
%non_robust_rand_inf100 = infid_grape_lanczos_T_robust(Jrand,Kxx,Mf,c0,ctg,bin_num,x_optm,numK)

%f = @(i) infid_grape_lanczos_T_robust(Jrand(i,:),Kxx,Mf,c0,ctg,bin_num,x_rob,numK);
%stdev = std(arrayfun(f, 1:101));

final_constraint = nonlcon(x_optm)
final_infidelity = fun(x_optm)
save(sprintf("out/dat_cluster_gs_robustZZ%d%d%g.mat", grad, n, dg))

end
end
