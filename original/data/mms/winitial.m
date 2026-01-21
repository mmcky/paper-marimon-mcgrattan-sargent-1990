% Initial data for Class00x.m, x=1,..4
%

%  Specify values for:
%
%    (1) maxit = number of iterations (each iteration involves all matchings)
%    (2) ntypes = number of types of agents
%    (3) nagents = a ntypes x 1 vector with nagents(i)=number of agents of 
%                 type i, i=1,...,ntypes
%    (4) nclasst = number of classifiers in trade classifier systems
%    (5) nclassc = number of classifiers in consume classifier systems
%    (6) bnames = a ntypes x l matrix of 0, 1, and -1's; each row is the 
%                 trinary expansion representation (length l) for an agent type
%    (7) rclass = 1 if classifier systems to be randomly generated
%                 0 if systems given as input;  if rclass=0, provide 2 
%                 matrices (CSt: nclasst x 2*l+1  and CSc: nclassc x l+1,
%                 i.e. conditions and actions) in "rules.m"
%    (8) produces = a ntypes x 1 vector with produces(i)= good type produced
%                   by type i
%    (9) storecosts = a ntypes x ntypes matrix with storecosts(i,j)= the cost
%                     to agent i of holding good j
%   (10) prodcosts = a ntypes x 1 vector with prodcosts(i)= the costs to
%                    agent i for producing a new good
%   (11) utility = a ntypes x 1 vector with utility(i)= the utility gain
%                  to agent type i for consuming good type i
%   (12) strengtht = a nclasst x ntypes matrix with initial strengths
%                    for the trade classifiers
%   (13) strengthc = a nclassc x ntypes matrix with initial strengths
%                    for the consume classifiers
%   (14) tbid1,tbid2 = ntypes x 1 vectors where the bid for a trade 
%                      classifier is:  tbid1+tbid2*sp, sp=1/(1+#)
%   (15) cbid1,cbid2 = ntypes x 1 vectors where the bid for a consume
%                      classifier is:  cbid1+cbid2*sp, sp=1/(1+#)
%   (16) Taxt = a nclasst x ntypes matrix with lump-sum taxes for CSt
%   (17) Taxc = a nclassc x ntypes matrix with lump-sum taxes for CSc
%   (18) probt = a 2 x 2*l matrix with probt(1,i)=probability of generating
%                a -1 for bit i of the strings of the trade classifiers,
%                i=1,...,2*l, l=length of own or match's storage
%                and probt(2,i)=probability of generating a 0 for the bit i
%   (19) probc = a 2 x l matrix with probc(1,i)=probability of generating
%                a -1 for bit i of the strings of the consume classifiers,
%                i=1,...,l, l=length of own or match's storage
%                and probc(2,i)=probability of generating a 0 for the bit i
%   (20) dclasst = number of periods between displays of trade classifier
%                  systems 
%   (21) dclassc = number of periods between displays of trade classifier
%                  systems 
%   (22) dhist = number of periods between displays of histograms
%   (23) dprob = number of periods between displays of probability matrices
%   (24) saveh = number of periods between storing histograms
%   (25) savec = number of periods between storing information on classifiers
%   (26) savef = number of periods between storing information on frequencies
%   (27) nback = number of periods used in computing statistics
%   (28) propselectt = the proportion of strings in a trade classifier
%                      system chosen for reproduction
%   (29) propselectt = the proportion of strings in a consume classifier
%                      system chosen for reproduction
%   (30) pcrosst = the probability of crossover in trade classifier system
%   (31) pcrossc = the probability of crossover in consume classifier system
%   (32) pmutationt = the probability of mutation in trade classifier 
%   (33) pmutationc = the probability of mutation in consume classifier 
%   (34) crowdsubpopt = the crowding subpopulation for trade system
%   (35) crowdsubpopc = the crowding subpopulation for consume system
%   (36) crowdfactort = the crowding factor for trade classifier system
%   (37) crowdfactorc = the crowding factor for consume classifier system
%   (38) propmostusedt = the proportion of most used rules in trade 
%                        classifier chosen as candidates in reproduction
%   (39) propmostusedc = the proportion of most used rules in consumer
%                        classifier chosen as candidates in reproduction
%   (40) runit = a maxit x 1 vector with runit(i)=1 if the genetic algorithm
%                is to be used at iteration i and 0 otherwise
%   (41) psecond = the probability that a second classifier system goes
%                  to the genetic algorithm given one already has
%   (42) pthird = the probability that a third classifier system goes
%                 to the genetic algorithm given two already have
%   (43) uratio = a 2 x 1 vector with criteria for rule elimination in
%                 genetic algorithm;  uratio(1)= cutoff on strength
%                 uratio(2)= cutoff with respect to #used/max(#used)
%

% parameters for describing classifier systems:

maxit=1000;
ntypes=3;
nagents=[50;50;50];
nclasst=72;
nclassc=12;
rclass=1;
bnames=[1 0 0;
        0 1 0;
        0 0 1];
produces=[2;3;1];
storecosts=[.1,1,20;
            .1,1,20;
            .1,1,20];
prodcosts=[1;1;1];
utility=[100;100;100];
strengtht=zeros(nclasst,ntypes);
strengthc=zeros(nclassc,ntypes);
tbid1=0.5*ones(ntypes,1);
tbid2=0.5*ones(ntypes,1);
cbid1=0.5*ones(ntypes,1);
cbid2=0.5*ones(ntypes,1);
Taxt=ones(nclasst,ntypes);
Taxc=ones(nclassc,ntypes);
probt=[.33 .33 .33 .33 .33 .33;
       .33 .33 .33 .33 .33 .33];
probc=[.33 .33 .33;
       .33 .33 .33];

% parameters for printing and saving information:

dclasst=100;
dclassc=100;
dhist=20;
dprob=5000;
saveh=20;
savec=50;
savef=250;
nback=10;

% parameters for genetic algorithm:

propselectt=.2;
propselectc=.2;
pcrosst=.6;
pcrossc=.6;
pmutationt=.01;
pmutationc=.01;
crowdfactort=8;
crowdfactorc=4;
crowdsubpopt=.5;
crowdsubpopc=.5;
propmostusedt=.7;
propmostusedc=.7;
rand;
pga=ones(maxit/2,1)./(sqrt([1:maxit/2]'));
runit=zeros(maxit,1);
runit(~rem([1:maxit]',2),1)=pga;
runit=(runit>rand(maxit,1));
psecond=.33;
pthird=.33;
uratio=[0;.2];
ufitness=.5;

