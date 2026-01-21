% Initial data for the Wicksell.m.


ntypes=3;
nagents=50;
nclassifiers=60;
bnames=[0 0;
        1 0;
       -1 1];
produces=[2;3;1];
storecosts=[.1,1,4;
            .1,1,4;
            .1,1,4];
prodcosts=[1;1;1];
utility=[20;20;20];
strength=ones(nclassifiers,ntypes);
maxit=90; 
bid1=0.1*ones(ntypes,1);
bid2=0.1*ones(ntypes,1);
tax=0.0001*ones(ntypes,1);

prob=[.33 3/12 .33 3/12;
      .33 7/12 .33 7/12];
lchrom=2;
dispclass=10;
Tga=10;

proportionselect=.4;
pcross=.6;
pmutation=.05;
crowdingfactor=20;
crowdingsubpop=20;
smultiple=2;
