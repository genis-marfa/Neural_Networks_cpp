%% Neural Networks:
% Main plot:
p = load('test_points_spiral.txt');
cl = p(:,3);
plot(p(cl==1,1), p(cl==1,2),'bx', 'MarkerSize',15);
hold on
plot(p(cl==-1,1), p(cl==-1,2),'ro', 'MarkerSize',15);
cdat = load('test_contour_spiral.txt');
contour(linspace(0,1,251),linspace(0,1,251), cdat',[0 0],'k--');
%% Convergence plots - Eta:
% Eta=0.1
iter= load('Num_Iter_eta10.txt');
n = (1:size(iter));
logiter=log(iter);
mean(iter)

% Eta=0.005
iter2=load('Num_Iter_eta05.txt');
logiter2=log(iter2);
mean(iter2)

% Eta=0.05
iter3=load('Num_Iter_eta_point5.txt');
logiter3=log(iter3);
mean(iter3)

% Eta=0.3
iter4=load('Num_Iter_eta30.txt');
n2 = [0, 3, 7, 8, 12, 14, 19, 22, 25, 31, 35, 39, 40, 44, 45, 47, 51, 53, 54, 59, 60, 61, 67, 71, ...
    73, 76, 78, 80, 83, 85, 88, 91, 95, 98, 99];
logiter4=log(iter4);
mean(iter4)

plot(n, logiter, 'x-r');
hold on
plot(n, logiter2, 'x-b');
hold on 
plot(n, logiter3, 'x-g');
hold on
plot(n2, logiter4, 'x-k');

%% Total Cost Plots - Eta:
Cost=readtable('Tot_Cost_eta_point1.txt');
Cost2=readtable('Tot_Cost_eta_point01.csv')
Cost3=readtable('Tot_Cost_eta_1.txt');

% Eta = 0.1
Iter1=Cost{:,1};
Totcost1=Cost{:,2};
logTotcost1=log(Totcost1);

% Eta = 0.01
Iter2=[1000:1000:208000];
Totcost2=Cost2{:,1};
logTotcost2=log(Totcost2);

% Eta = 1
Iter3=Cost3{:,1};
Totcost3=Cost3{:,2};
logTotcost3=log(Totcost3);

Iter3=Iter3(1:250);
logTotcost3=logTotcost3(1:250);

plot(Iter1, logTotcost1, 'x-r')
hold on
plot(Iter2, logTotcost2, 'x-k')
hold on
plot(Iter3, logTotcost3, 'x-b')
hold on
line([0,250000],[log(1e-4),log(1e-4)])
ylim([-10 0.5])

%% Total Cost Plots - initsd:
% Fix eta at 0.1, test initsd for 0.01, 0.1, 1.0, 5.0

Cost=readtable('Tot_Cost_eta_point1.txt'); % This corresponds to the 0.1 case. 
Cost2=readtable('Tot_Cost_initsd_poinnt01.txt');
Cost3=readtable('Tot_Cost_initsd_1.txt');
Cost4=readtable('Tot_Cost_initsd_5.txt');

% Initsd = 0.1:
Iter1=Cost{:,1};
Totcost1=Cost{:,2};
logTotcost1=log(Totcost1);

% Initsd = 0.01:
Iter2=Cost2{:,1};
Totcost2=Cost2{:,2};
logTotcost2=log(Totcost2);

% Initsd = 1.0:
Iter3=Cost3{:,1}
Totcost3=Cost3{:,2};
logTotcost3=log(Totcost3);

% Initsd = 5.0:
Iter4=Cost4{:,1}
Totcost4=Cost4{:,2};
logTotcost4=log(Totcost4);

plot(Iter1, logTotcost1, 'x-r')
hold on
plot(Iter2, logTotcost2, 'x-k')
hold on
%plot(Iter3, logTotcost3, 'x-b')
%hold on
plot(Iter4, logTotcost4, 'x-m')
hold on
line([0,250000],[log(1e-4),log(1e-4)])
xlim([0 100000])
ylim([-10 0.5])

%% Convergence plots - initsd:
% Initsd=0.1
iter= load('Num_Iter_initsd_point1.txt')
n = (1:size(iter));
logiter=log(iter);
mean(iter)

% Initsd=0.01
iter2=load('Num_Iter_initsd_point01.txt');
logiter2=log(iter2);
mean(iter2)

% Initsd=1
iter3=load('Num_Iter_initsd_1.txt');
logiter3=log(iter3);
mean(iter3)

% Initsd=3
iter4=[18000, 31000, 316000, 210000, 286000, 16000, 59000, 311000, 217000, 47000, 29000, ...
30000, 96000, 86000, 891000, 100000, 164000, 185000, 51000, 25000, 927000, 47000];
n_4  =[4, 5, 6, 7, 10, 11, 13, 15, 19, 21, 22, 28, 29, 30, 32, 35, 37, ...
       41, 42, 45, 48, 49];

logiter4=log(iter4);
mean(iter4)

plot(n, logiter, 'x-r');
hold on
plot(n, logiter2, 'x-k');
hold on 
%plot(n, logiter3, 'x-b');
%hold on
plot(n_4, logiter4, 'x-m');
hold on
