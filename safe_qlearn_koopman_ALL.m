function safe_qlearn_koopman_ALL()
%% ========================================================================
%  safe_qlearn_koopman_ALL.m   (single self-contained file)
%
%  Model-Free, Optimal, and Safe Q-Learning in Koopman Eigenfunction
%  Coordinates (Dony, Qureshi, Rizvi).  Implements the full consolidated
%  derivation and reproduces BOTH 6-panel figures:  Example 1 (linear car),
%  Example 2 (nonlinear Vaidya).
%
%  Every math line is annotated with its equation number (Parts A-L).  Tags:
%    [J] Jha-Kiumarsi (CBF/safety)  [V] Vaidya (eigenfunction optimal control)
%    [R] Rizvi-Lin (model-free Q-learning)  [Ours] this paper.
%
%  The learning core (qlearn_lspi) is FULLY MODEL-FREE: only measured (z,u,z+)
%  and the running cost -- never f,g,A,B, or plant eigen-data.
%
%  Requirements: base MATLAB, R2018b+ (uses yline/xline/sgtitle). NO toolboxes
%  (the DARE reference is solved by Riccati iteration).
%
%  Run:  safe_qlearn_koopman_ALL
%  Out:  example1_car_matlab.pdf/.png , example2_vaidya_matlab.pdf/.png
%        (PDFs are tight vector crops -> include at \textwidth without margin
%         overflow; see save_fig at the bottom of this file.)
%% ========================================================================
    run_car();
    run_vaidya();
    disp('Done: example1_car_matlab.png, example2_vaidya_matlab.png');
end


%% ========================================================================
%  EXAMPLE 1 -- LINEAR CAR LANE-KEEPING  [J(65)]
%  Trivial lift Phi(x)=x => framework reduces to model-free Q-learning [R]+CBF.
%% ========================================================================
function run_car()
    rng(1);
    % --- plant (eq ex1sys, ex1disc): lateral bicycle model, Euler discretized -
    M_=1650; Iz=2315.3; v0=27.7; Cf=133000; Cr=98800; af=1.11; br=1.59; Ts=0.01;
    Ac = [0 1 v0 0;
          0 -(Cf+Cr)/(M_*v0) 0 (br*Cr-af*Cf)/(M_*v0)-v0;
          0 0 0 1;
          0 (br*Cr-af*Cf)/(Iz*v0) 0 -(af^2*Cf+br^2*Cr)/(Iz*v0)];
    Bc = [0; Cf/M_; 0; af*Cf/Iz];
    A = eye(4)+Ts*Ac; B = Ts*Bc; n=4; m=1;                     % (ex1disc)

    Q=eye(4); R=1; ymax=0.5; gamma=0.95; beta=1.7;
    Afaces=[1 0 0 0; -1 0 0 0]; cfaces=[ymax;ymax];            % h1=x1+ymax, h2=ymax-x1
    HB = barrier_hessian(Afaces,cfaces,gamma,beta);            % (F15) expect 12.19
    Phi = @(x) x(:);                                           % trivial lift z=Phi(x)=x
    fprintf('[car] H_B(1,1)=%.3f (exp 12.19)\n',HB(1,1));

    % --- model-free seed (Phase-0 opt (i)): estimate [A,B] from a burst -> DARE -
    Lb=400; xb=zeros(4,1); Xb=zeros(Lb,4); Ub=zeros(Lb,1); Xbp=zeros(Lb,4);
    for k=1:Lb, ub=0.5*randn; xn=A*xb+B*ub; Xb(k,:)=xb'; Ub(k)=ub; Xbp(k,:)=xn'; xb=xn; end
    Reg=[Xb Ub]; W=(Xbp'*Reg)*pinv(Reg'*Reg);
    [~,K0]=dare_gain(W(:,1:4),W(:,5),Q,R);                     % stabilizing seed

    % --- data collection (Phase 2): u=K0 x+exploration (small => interior) -----
    L=800; freqs=linspace(0.5,40,50); amps=0.012*randn(1,50);
    x=[0.05;0;0;0]; Xk=zeros(L,4); Uk=zeros(L,1); Xk1=zeros(L,4);
    for k=1:L
        u=K0*x+sum(amps.*sin(freqs*(k-1)*Ts)); xn=A*x+B*u;    % (J5) PE input
        Xk(k,:)=x'; Uk(k)=u; Xk1(k,:)=xn'; x=xn;
    end
    Zk=Xk; Zk1=Xk1;
    fprintf('[car] max|x1| exploration=%.3f (interior)\n',max(abs(Xk(:,1))));

    % --- safe Q-learning (eq J1-J6): two runs on the SAME data ----------------
    Rs_std=zeros(L,1); Rs_cbf=zeros(L,1);
    for k=1:L
        perf=Xk(k,:)*Q*Xk(k,:)'+R*Uk(k)^2;                    % (B5) performance
        Rs_std(k)=perf;
        Rs_cbf(k)=perf+barrier_value(Xk(k,:)',Afaces,cfaces,gamma,beta);
    end
    [K_std,~,Kh_std]=qlearn_lspi(Zk,Uk,Zk1,Rs_std,K0,n,m,6);
    [K_cbf,H_cbf,Kh_cbf]=qlearn_lspi(Zk,Uk,Zk1,Rs_cbf,K0,n,m,6);
    [~,Kref_cbf]=dare_gain(A,B,Q+0.5*HB,R);
    fprintf('[car] K_std=[%s]\n',sprintf('%.3f ',K_std));
    fprintf('[car] K_cbf=[%s] ref [%s]\n',sprintf('%.3f ',K_cbf),sprintf('%.3f ',Kref_cbf));

    % --- certified region (eq L1): closed-form Schur complement (linear lift) --
    IK=[eye(n);K_cbf]; Pbar=IK'*H_cbf*IK; Pbar=0.5*(Pbar+Pbar');
    schur=Pbar(1,1)-Pbar(1,2:end)*(Pbar(2:end,2:end)\Pbar(2:end,1));
    cstar=ymax^2*schur;
    fprintf('[car] Pbar(1,1)=%.1f c*=%.1f radius %.3f\n',Pbar(1,1),cstar,sqrt(cstar/Pbar(1,1)));

    % --- closed-loop simulation  u=K Phi(x) ----------------------------------
    sim=@(K,x0,T) sim_cl(K,x0,T,A,B,Ts);
    IC={[0.20;0.8;0.05;0.01],[-0.25;-0.8;-0.03;-0.01],[0.15;1.2;0.08;0.03],[0.10;-1.0;0.05;0.02]};

    % --- figure (6 panels) ---------------------------------------------------
    figure('Units','inches','Position',[1 1 11 9.2],'Color','w');
    set(gcf,'DefaultAxesFontSize',12);                        % legible at \textwidth
    sgtitle({'Safe Model-Free Q-Learning in Eigenfunction Coordinates','Linear System - Car Lane Keeping'},'FontWeight','bold');

    [t,xc,~]=sim(K_cbf,IC{3},10); [~,xs,~]=sim(K_std,IC{3},10);
    subplot(3,2,1); hold on; grid on;
    patch([0 10 10 0],[ymax ymax 0.9 0.9],[1 .9 .9],'EdgeColor','none','HandleVisibility','off');
    patch([0 10 10 0],[-0.9 -0.9 -ymax -ymax],[1 .9 .9],'EdgeColor','none','HandleVisibility','off');
    hc=plot(t,xc(:,1),'b-','LineWidth',2); hs=plot(t,xs(:,1),'r--','LineWidth',2);
    hl=yline(ymax,'Color',[.5 0 .5],'LineWidth',3); yline(-ymax,'Color',[1 .6 0],'LineWidth',3);
    ylim([-0.9 0.9]); xlabel('Time (s)'); ylabel('x_1 (lateral displacement) [m]');
    title('(a) CBF Q-Learning vs Standard Q-Learning (IC3)');
    legend([hc hs hl],{'x_1 with CBF (our method)','x_1 without CBF (standard)','Safety limit \pm0.5 m'},'FontSize',7);

    subplot(3,2,2); hold on; grid on; safe=true;
    for a=[-.25 -.1 .1 .25], for b=[-1 1], for c=[-.05 .05]
        [~,xx,~]=sim(K_cbf,[a;b;c;0],10); plot(t,xx(:,1),'LineWidth',.8,'HandleVisibility','off');
        if max(abs(xx(:,1)))>ymax, safe=false; end
    end; end; end
    h1=yline(ymax,'r--','LineWidth',2.5); h2=yline(-ymax,'c--','LineWidth',2.5);
    xlabel('Time (s)'); ylabel('x_1'); title(sprintf('(b) Multiple Initial Conditions (all safe: %d)',safe));
    legend([h1 h2],{'y_{max}=0.5 m','-y_{max}'},'FontSize',8);

    [t1,x1s,~]=sim(K_cbf,IC{1},5);
    subplot(3,2,3); hold on; grid on;
    g1=plot(t1,x1s(:,1),'b-','LineWidth',1.6); g2=plot(t1,x1s(:,2),'g--','LineWidth',1.6);
    g3=plot(t1,x1s(:,3),'r:','LineWidth',1.6); g4=plot(t1,x1s(:,4),'m-.','LineWidth',1.6);
    xlabel('Time (s)'); ylabel('State value'); title('(c) State Trajectories (CBF policy, IC1)');
    legend([g1 g2 g3 g4],{'x_1 (lateral disp)','x_2 (lateral vel)','x_3 (yaw angle err)','x_4 (yaw rate)'},'FontSize',7);

    [t3,~,uc3]=sim(K_cbf,IC{3},4); [~,~,us3]=sim(K_std,IC{3},4);
    subplot(3,2,4); hold on; grid on;
    d1=plot(t3,uc3,'b-','LineWidth',1.4); d2=plot(t3,us3,'r--','LineWidth',1.4);
    xlabel('Time (s)'); ylabel('Steering angle u'); title('(d) Control Input Comparison (IC3)');
    legend([d1 d2],{'CBF Q-learning','Standard Q-learning'},'FontSize',8);

    subplot(3,2,5); hold on; grid on; mk={'bo-','gs-','r^-','md-'}; hh=gobjects(1,4);
    for i=1:4, hh(i)=plot(0:size(Kh_cbf,1)-1,Kh_cbf(:,i),mk{i},'LineWidth',1.2); end
    xlabel('Iteration j'); ylabel('Gain value'); title('(e) Policy Gain Convergence (CBF Q-Learning)');
    legend(hh,{'K_1 (lateral)','K_2 (lat vel)','K_3 (yaw)','K_4 (yaw rate)'},'FontSize',7,'Location','southwest');

    subplot(3,2,6); hold on; grid on;
    Jc=zeros(1,size(Kh_cbf,1));                        % CBF and standard runs may
    for j=1:size(Kh_cbf,1)                             % converge in different #iters
        Jc(j)=total_cost_car(Kh_cbf(j,:),true ,IC{3},A,B,Ts,Q,R,Afaces,cfaces,gamma,beta);
    end
    Js=zeros(1,size(Kh_std,1));                        % => loop each to its own length
    for j=1:size(Kh_std,1)
        Js(j)=total_cost_car(Kh_std(j,:),false,IC{3},A,B,Ts,Q,R,Afaces,cfaces,gamma,beta);
    end
    f1=semilogy(0:numel(Jc)-1,Jc,'bo-','LineWidth',1.3); f2=semilogy(0:numel(Js)-1,Js,'rs--','LineWidth',1.3);
    xlabel('Iteration j'); ylabel('Total simulated cost'); title('(f) Cost Convergence');
    legend([f1 f2],{'CBF cost \Sigma r_s','Standard cost \Sigma r'},'FontSize',8);

    save_fig(gcf,'example1_car_matlab'); fprintf('[car] figure saved (pdf+png).\n\n');
end


%% ========================================================================
%  EXAMPLE 2 -- NONLINEAR SYSTEM (VAIDYA)  [V(94)]
%  Full pipeline; learned policy u=K1(x1-2x2)+K2(x1+sin x2) is NONLINEAR in x.
%% ========================================================================
function run_vaidya()
    rng(3); Ts=0.01; n=2; m=1;
    % --- true plant (eq ex2drift): continuous drift + RK4 ZOH; eigfns (ex2eig) -
    F=@(x)(1/(cos(x(2))+2))*[ -cos(x(2))*(x(1)-2*x(2))+4*(x(1)+sin(x(2)));
                              (x(1)-2*x(2))+2*(x(1)+sin(x(2))) ];
    step=@(x,u) rk4(x,u,Ts,F);
    phi1t=@(x) x(1)-2*x(2); phi2t=@(x) x(1)+sin(x(2));
    psi=@(x)[ sin(x(2))-x(2); x(2)^2; x(1)*x(2); x(2)^3; ...
              x(1)*sin(x(2)); x(2)*sin(x(2)); x(1)^2; x(1)*x(2)^2 ];   % (K1)

    Qz=eye(2); Rval=1; gamma=1.0; beta=2.0;
    Afaces=[1 0; -1 0]; cfaces=[0.55;0.55];
    HB=barrier_hessian(Afaces,cfaces,gamma,beta);                     % (F15) expect 11.56
    fprintf('[vaidya] H_B(1,1)=%.3f (exp 11.56)\n',HB(1,1));

    % --- Phase 0 (eq K2-K3): uncontrolled short rollouts (unstable mode +2) ----
    Xu=[]; Xup=[];
    for r=1:1000
        x=(rand(2,1)-0.5)*3;
        for s=1:12
            xn=step(x,0);
            if max(abs(xn))<2.5, Xu=[Xu;x']; Xup=[Xup;xn']; end %#ok<AGROW>
            x=xn; if max(abs(x))>2.5, break; end
        end
    end
    [~,~,M_hat,V_hat]=phase0_spectral(Xu,Xup,psi);
    fprintf('[vaidya] mu=[%.6f %.6f] (true %.6f,%.6f)\n',diag(M_hat),exp(-Ts),exp(2*Ts));
    fprintf('[vaidya] V^T=[%.3f %.3f;%.3f %.3f] (true [1 -2;1 1])\n',V_hat);

    % --- Phase 1 (eq K5): eigenfunction learning -----------------------------
    [~,Phi]=phase1_eigfun(Xu,Xup,M_hat,V_hat,psi);
    e2=0; for a=linspace(-1,1,50), z=Phi([a;0.3]); e2=max(e2,abs(z(2)-phi2t([a;0.3]))); end
    fprintf('[vaidya] Phase-1 phi2 max error=%.4f (<1%%)\n',e2);

    % --- model-free seed (eq K4): estimate Btilde from a burst, then DARE ------
    xb=[0.1;0]; Ub=[]; dZ=[];
    for k=1:300
        ub=0.3*randn; xn=step(xb,ub);
        Ub=[Ub;ub]; dZ=[dZ;(Phi(xn)-M_hat*Phi(xb))']; xb=xn; %#ok<AGROW>
        if max(abs(xb))>1.5, xb=[0.1;0]; end
    end
    Btil=(dZ'*Ub)/(Ub'*Ub);                                          % (K4)
    fprintf('[vaidya] Btilde=[%.6f %.6f] (ZOH ~[0.009950 0.010101])\n',Btil);
    [~,K0]=dare_gain(M_hat,Btil,Qz,Rval);
    Qg=V_hat\HB/(V_hat');                                            % (F16) Q_gamma
    [~,Kref_cbf]=dare_gain(M_hat,Btil,Qz+Qg,1);

    % --- Phase 2 (eq J1-J6): data + safe Q-learning; r_s=1/2 z'z+1/2 u^2 [+B] --
    L=3000; freqs=linspace(0.5,30,40); amps=0.05*randn(1,40);
    x=[0.05;0]; Xk=zeros(L,2); Uk=zeros(L,1); Xk1=zeros(L,2);
    for k=1:L
        u=K0*Phi(x)+sum(amps.*sin(freqs*(k-1)*Ts)); xn=step(x,u);
        Xk(k,:)=x'; Uk(k)=u; Xk1(k,:)=xn'; x=xn;
        if max(abs(x))>1.5, x=[0.05;0]; end
    end
    Zk=zeros(L,2); Zk1=zeros(L,2);
    for k=1:L, Zk(k,:)=Phi(Xk(k,:)')'; Zk1(k,:)=Phi(Xk1(k,:)')'; end
    fprintf('[vaidya] max|x1| exploration=%.3f (interior)\n',max(abs(Xk(:,1))));
    Rs_std=zeros(L,1); Rs_cbf=zeros(L,1);
    for k=1:L
        perf=0.5*(Zk(k,:)*Qz*Zk(k,:)')+0.5*Rval*Uk(k)^2;            % (B5) 1/2-convention
        Rs_std(k)=perf;
        Rs_cbf(k)=perf+barrier_value(Xk(k,:)',Afaces,cfaces,gamma,beta);
    end
    [K_std,~,~]     =qlearn_lspi(Zk,Uk,Zk1,Rs_std,K0,n,m,8);
    [K_cbf,H_cbf,Kh_cbf]=qlearn_lspi(Zk,Uk,Zk1,Rs_cbf,K0,n,m,8);
    fprintf('[vaidya] K_std=[%s]\n',sprintf('%.3f ',K_std));
    fprintf('[vaidya] K_cbf=[%s] ref [%s]\n',sprintf('%.3f ',K_cbf),sprintf('%.3f ',Kref_cbf));

    % --- certified region (eq L1): grid the faces x1=+/-0.55 -----------------
    bpts=[0.55*ones(4001,1) linspace(-2,2,4001)'; -0.55*ones(4001,1) linspace(-2,2,4001)'];
    [Pbar,cstar]=certify_region(H_cbf,K_cbf,Phi,bpts,n);
    z10=Phi([1;0]); rad=sqrt(cstar/(z10'*Pbar*z10));
    fprintf('[vaidya] Pbar=[%.2f %.2f;%.2f %.2f] c*=%.1f radius %.3f\n',Pbar',cstar,rad);

    % --- closed-loop sim  u=K Phi(x) (true nonlinear plant, RK4) -------------
    sim=@(K,x0,T) sim_nl(K,x0,T,step,Phi);
    IC={[0.3;0.8],[-0.3;-0.8],[0.3;1.0],[0.4;0.8],[-0.4;0.6]}; xmax=0.55;
    col={'b','r','g',[1 .6 0],[.5 0 .5]};

    % --- figure (6 panels) --------------------------------------------------
    figure('Units','inches','Position',[1 1 11 9.2],'Color','w');
    set(gcf,'DefaultAxesFontSize',12);                        % legible at \textwidth
    sgtitle({'Safe Model-Free Q-Learning in Eigenfunction Coordinates','Nonlinear System'},'FontWeight','bold');

    [t,xc,~]=sim(K_cbf,IC{3},15); [~,xs,~]=sim(K_std,IC{3},15);
    subplot(3,2,1); hold on; grid on;
    patch([0 15 15 0],[xmax xmax 0.9 0.9],[1 .9 .9],'EdgeColor','none','HandleVisibility','off');
    patch([0 15 15 0],[-0.9 -0.9 -xmax -xmax],[1 .9 .9],'EdgeColor','none','HandleVisibility','off');
    hc=plot(t,xc(:,1),'b-','LineWidth',2); hs=plot(t,xs(:,1),'r--','LineWidth',2);
    hl=yline(xmax,'Color',[.5 0 .5],'LineWidth',3); yline(-xmax,'Color',[1 .6 0],'LineWidth',3);
    ylim([-0.9 0.9]); xlabel('Time (s)'); ylabel('x_1'); title('(a) CBF vs Standard Q-Learning');
    legend([hc hs hl],{'x_1 with CBF (our method)','x_1 without CBF (standard)','Safety limit \pm0.55'},'FontSize',7);

    subplot(3,2,2); hold on; grid on; hh=gobjects(1,5);
    for i=1:5
        [~,xx,~]=sim(K_cbf,IC{i},15); hh(i)=plot(xx(:,1),xx(:,2),'Color',col{i},'LineWidth',1.5);
        plot(IC{i}(1),IC{i}(2),'ko','MarkerFaceColor','k','MarkerSize',5,'HandleVisibility','off');
    end
    xline(xmax,'r--','HandleVisibility','off'); xline(-xmax,'r--','HandleVisibility','off');
    xlabel('x_1'); ylabel('x_2'); title('(b) Phase Portrait (CBF)');
    legend(hh,{'IC1','IC2','IC3','IC4','IC5'},'FontSize',7);

    xr=linspace(-1,1,60); p1t=zeros(60,1); p1l=zeros(60,1); p2t=zeros(60,1); p2l=zeros(60,1);
    for i=1:60, xx=[xr(i);0.3]; z=Phi(xx); p1t(i)=phi1t(xx); p1l(i)=z(1); p2t(i)=phi2t(xx); p2l(i)=z(2); end
    subplot(3,2,3); hold on; grid on;
    c1=plot(xr,p1t,'b-','LineWidth',2); c2=plot(xr,p1l,'b--','LineWidth',2);
    c3=plot(xr,p2t,'r-','LineWidth',2); c4=plot(xr,p2l,'r--','LineWidth',2);
    xlabel('x_1'); title('(c) Learned vs True Eigenfunctions (x_2=0.3)');
    legend([c1 c2 c3 c4],{'\phi_1 true','\phi_1 learned','\phi_2 true','\phi_2 learned'},'FontSize',7);

    [t5,~,uc5]=sim(K_cbf,IC{5},8); [~,~,us5]=sim(K_std,IC{5},8);
    [t3,~,uc3]=sim(K_cbf,IC{3},8); [~,~,us3]=sim(K_std,IC{3},8);
    subplot(3,2,4); hold on; grid on;
    d1=plot(t5,uc5,'b-','LineWidth',1.6); d2=plot(t5,us5,'r--','LineWidth',1.6);
    d3=plot(t3,uc3,'-','LineWidth',.8,'Color',[.4 .4 1]); d4=plot(t3,us3,'--','LineWidth',.8,'Color',[1 .5 .5]);
    xlabel('Time (s)'); ylabel('Control u'); title('(d) Control Input Comparison');
    legend([d1 d2 d3 d4],{'CBF, IC5','Standard, IC5','CBF, IC3','Standard, IC3'},'FontSize',7);

    subplot(3,2,5); hold on; grid on;
    e1=plot(0:size(Kh_cbf,1)-1,Kh_cbf(:,1),'bo-','LineWidth',1.3);
    e2h=plot(0:size(Kh_cbf,1)-1,Kh_cbf(:,2),'rs-','LineWidth',1.3);
    xlabel('Iteration j'); ylabel('Gain'); title('(e) Gain Convergence (CBF)');
    legend([e1 e2h],{'K_1','K_2'},'FontSize',8);

    subplot(3,2,6); hold on; grid on; safe=true;
    for a=linspace(-0.32,0.35,8), for b=linspace(-0.35,0.35,3)
        [tt,xx,~]=sim(K_cbf,[a;b],15); plot(tt,xx(:,1),'LineWidth',.8,'HandleVisibility','off');
        if max(abs(xx(:,1)))>xmax, safe=false; end
    end; end
    yline(xmax,'r--','LineWidth',2.5,'HandleVisibility','off'); yline(-xmax,'c--','LineWidth',2.5,'HandleVisibility','off');
    xlabel('Time (s)'); ylabel('x_1'); title(sprintf('(f) Multiple ICs (all safe: %d)',safe));

    save_fig(gcf,'example2_vaidya_matlab'); fprintf('[vaidya] figure saved (pdf+png).\n\n');
end


%% ========================================================================
%  CORE FRAMEWORK  (model-free; equation-referenced)
%% ========================================================================
function B = barrier_value(x, Afaces, cfaces, gamma, beta)
% (B3-B4) recentered CBF penalty  B_gamma(x)=beta sum_i [b_i(x)-b_i(0)],
%         h_i=c_i+a_i^T x,  b(h)=-log(gamma h/(gamma h+1)).  +Inf outside S.
    tot=0.0;
    for i=1:size(Afaces,1)
        h=cfaces(i)+Afaces(i,:)*x(:);
        if h<=1e-9, B=Inf; return; end
        b =-log(gamma*h /(gamma*h +1));            % (B3) at x
        b0=-log(gamma*cfaces(i)/(gamma*cfaces(i)+1)); % (B3) at origin
        tot=tot+(b-b0);                            % (B4) recentered
    end
    B=beta*tot;
end

function HB = barrier_hessian(Afaces, cfaces, gamma, beta)
% (F15) H_B=beta sum_i b''(c_i) a_i a_i^T, b''(h)=(2 gamma h+1)/(h(gamma h+1))^2
    n=size(Afaces,2); HB=zeros(n,n);
    for i=1:size(Afaces,1)
        c=cfaces(i); bpp=(2*gamma*c+1)/(c*(gamma*c+1))^2;
        HB=HB+beta*bpp*(Afaces(i,:)'*Afaces(i,:));
    end
end

function [A_hat,N_hat,M_hat,V_hat] = phase0_spectral(Xu, Xup, psi)
% (K2-K3) joint LS x+=A_d x+N psi(x); eig(A_d^T)->M=diag(mu), V=left eigvecs.
    n=size(Xu,2); L=size(Xu,1); Psi=zeros(L,numel(psi(Xu(1,:)')));
    for k=1:L, Psi(k,:)=psi(Xu(k,:)')'; end
    Chi=[Xu Psi]; W=(Xup'*Chi)*pinv(Chi'*Chi);     % (K2)
    A_hat=W(:,1:n); N_hat=W(:,n+1:end);
    [Vr,D]=eig(A_hat'); mu=real(diag(D)); [mu,idx]=sort(mu); Vr=real(Vr(:,idx));  % (K3)
    for i=1:n, Vr(:,i)=Vr(:,i)/Vr(1,i); end        % normalize first entry -> 1
    M_hat=diag(mu); V_hat=Vr;
end

function [Theta,Phi] = phase1_eigfun(Xu, Xup, M_hat, V_hat, psi)
% (K5) Phi_hat(x)=V^T x+Theta psi(x); per-row LS (M diagonal decouples).
    n=size(Xu,2); L=size(Xu,1); Nb=numel(psi(Xu(1,:)'));
    Psi=zeros(L,Nb); Psip=zeros(L,Nb);
    for k=1:L, Psi(k,:)=psi(Xu(k,:)')'; Psip(k,:)=psi(Xup(k,:)')'; end
    mu=diag(M_hat); Theta=zeros(n,Nb);
    for i=1:n
        vi=V_hat(:,i);
        Rrow=Psip-mu(i)*Psi;                       % (K5) psi(x+)-mu psi(x)
        targ=mu(i)*(Xu*vi)-(Xup*vi);               % (K5) mu v^T x - v^T x+
        Theta(i,:)=(Rrow\targ)';
    end
    Phi=@(x) V_hat'*x(:)+Theta*psi(x(:));
end

function phi = quad_features(z)
% (J2) monomials z_i z_j (i<=j) so z'Hz = hbar . quad_features(z)
    l=numel(z); phi=zeros(l*(l+1)/2,1); idx=1;
    for i=1:l, for j=i:l, phi(idx)=z(i)*z(j); idx=idx+1; end; end
end

function H = unpack_H(hbar, l)
% (J2) inverse: rebuild symmetric H (halve off-diagonals)
    H=zeros(l,l); idx=1;
    for i=1:l, for j=i:l
        if i==j, H(i,i)=hbar(idx); else, H(i,j)=0.5*hbar(idx); H(j,i)=0.5*hbar(idx); end
        idx=idx+1;
    end; end
end

function [K,H,Khist] = qlearn_lspi(Zk, Uk, Zk1, Rs, K0, n, m, iters)
% (J1-J6, I10) model-free safe Q-learning by LSPI.  Data + cost only.
    if nargin<8, iters=8; end
    l=n+m; L=size(Zk,1); K=K0; Khist=K(:)'; H=[]; tol=1e-9;
    for it=1:iters
        Phi_data=zeros(L,l*(l+1)/2);
        for k=1:L
            zeta_k=[Zk(k,:)';Uk(k,:)'];                          % zeta_k=[z_k;u_k]
            zeta_k1=[Zk1(k,:)';K*Zk1(k,:)'];                     % (J1) u_{k+1}=K z_{k+1}
            Phi_data(k,:)=(quad_features(zeta_k)-quad_features(zeta_k1))';  % (J3)
        end
        hbar=Phi_data\Rs;                                        % (J4) LS solve
        H=unpack_H(hbar,l);
        Knew=-(H(n+1:end,n+1:end)\H(n+1:end,1:n));               % (I10) K=-Huu^-1 Huz
        Khist=[Khist;Knew(:)']; %#ok<AGROW>
        if norm(Knew-K)<tol, K=Knew; break; end
        K=Knew;
    end
end

function [Pbar,cstar] = certify_region(H, K, Phi, bpts, n)
% (L1-L2) Pbar=[I;K]'H[I;K]; c*=min_{x in dS} Phi(x)'Pbar Phi(x)
    IK=[eye(n);K]; Pbar=IK'*H*IK; Pbar=0.5*(Pbar+Pbar');
    nb=size(bpts,1); vals=zeros(nb,1);
    for k=1:nb, z=Phi(bpts(k,:)'); vals(k)=z'*Pbar*z; end
    cstar=min(vals);
end

function [P,K] = dare_gain(A, B, Q, R)
% DARE reference (VALIDATION/seed only -- uses the model).  Riccati iteration
% (mirrors model-free value iteration, eq 2.42); no Control System Toolbox.
    P=Q;
    for it=1:100000
        BtP=B'*P; Pnew=Q+A'*P*A-A'*P*B/(R+BtP*B)*(BtP*A); Pnew=0.5*(Pnew+Pnew');
        if norm(Pnew-P,'fro')<1e-12*max(1,norm(P,'fro')), P=Pnew; break; end
        P=Pnew;
    end
    K=-(R+B'*P*B)\(B'*P*A);
end


%% ========================================================================
%  SIMULATION HELPERS
%% ========================================================================
function [t,xs,us] = sim_cl(K, x0, T, A, B, Ts)
% linear closed loop  x+ = A x + B u,  u = K Phi(x) = K x
    N=round(T/Ts); xs=zeros(N,numel(x0)); us=zeros(N,1); x=x0(:);
    for k=1:N, u=K*x; us(k)=u; xs(k,:)=x'; x=A*x+B*u; end
    t=(0:N-1)'*Ts;
end

function J = total_cost_car(K, cbf, x0, A, B, Ts, Q, R, Afaces, cfaces, gamma, beta)
% Sigma r_s under gain K (barrier capped at boundary so unsafe seed shows high cost)
    [~,xs,us]=sim_cl(K,x0,10,A,B,Ts); J=0;
    for k=1:numel(us)
        c=xs(k,:)*Q*xs(k,:)'+R*us(k)^2;
        if cbf
            b=barrier_value(xs(k,:)',Afaces,cfaces,gamma,beta);
            if ~isfinite(b), b=1e5; end
            c=c+b;
        end
        J=J+c;
    end
end

function xn = rk4(x, u, Ts, F)
% RK4 with zero-order-hold input u
    g=@(z) F(z)+[u;0];
    k1=g(x); k2=g(x+Ts/2*k1); k3=g(x+Ts/2*k2); k4=g(x+Ts*k3);
    xn=x+Ts/6*(k1+2*k2+2*k3+k4);
end

function [t,xs,us] = sim_nl(K, x0, T, step, Phi)
% nonlinear closed loop  x+ = step(x,u),  u = K Phi(x)
    Ts=0.01; N=round(T/Ts); xs=zeros(N,2); us=zeros(N,1); x=x0(:);
    for k=1:N, u=K*Phi(x); us(k)=u; xs(k,:)=x'; x=step(x,u); end
    t=(0:N-1)'*Ts;
end

function save_fig(fig, name)
% SAVE_FIG  Write FIG as a tight, MARGIN-SAFE VECTOR PDF (+ a PNG preview).
%   exportgraphics with ContentType='vector' crops the PDF page to the figure
%   content (no letter/A4 page padding), so in LaTeX
%       \includegraphics[width=\textwidth]{NAME.pdf}
%   scales the figure exactly to the text block and it never crosses the margin.
%   (For a two-column class such as elsarticle, use \includegraphics
%   [width=\columnwidth]{NAME.pdf} inside a single-column figure, or
%   \includegraphics[width=\textwidth]{NAME.pdf} inside figure*.)
    set(fig,'Color','w');
    if exist('exportgraphics','file') == 2                    % R2020a and newer
        exportgraphics(fig,[name '.pdf'],'ContentType','vector','BackgroundColor','white');
        exportgraphics(fig,[name '.png'],'Resolution',200,'BackgroundColor','white');
    else                                                      % older-MATLAB fallback
        set(fig,'Units','inches'); p = get(fig,'Position');   % page = figure size
        set(fig,'PaperUnits','inches','PaperPositionMode','manual', ...
                'PaperSize',[p(3) p(4)],'PaperPosition',[0 0 p(3) p(4)]);
        print(fig,[name '.pdf'],'-dpdf','-painters');         % vector, tight page
        print(fig,[name '.png'],'-dpng','-r200');
    end
end
