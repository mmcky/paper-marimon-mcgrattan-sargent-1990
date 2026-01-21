function [xbest,fbest,gbest,code]=sga(func,lb,ub)
%SGA   A simple genetic algorithm for optimization with simple bounds.
%      [x,f,g,code]=sga(func,lb,ub) solves the following problem:
%     
%               max  f(x)       subject to  lb<= x <= ub
%                x
%      where func is a string giving the name of the .m file containing 
%      code for the function f in (*).
%
%      In addition to returning the optimal x vector, f(x), and g(x)=
%      df(x)/dx (or, if gradchk=0, g=0), SGA returns "code" where:
%
%      code      =  1   if norm(g)/(1+f(x))  < gtol (i.e. x optimal)
%      code      =  2   if the iteration limit (maxgen) is reached
%      (Note: the user can specify tolerances and other parameters
%       by editing this file.)

%
%      Ellen R. McGrattan, 11-01-88
%      Revised, 6-22-89    ERM
%
%      References:
%      ----------  
%      [1] Goldberg, David, GENETIC ALGORITHMS FOR OPTIMIZATION, SEARCH, AND
%            MACHINE LEARNING, (Menlo Park, CA: Addison-Wesley Co.), 1989.
%

% NOTE: The parameters for SGA are coded as binary strings and parameter
%       vectors are the strings for all elements of x linked together.
%       The length of the string for x(i) is user-defined and given by
%       lparam(i), i=1, ..., nparam.  For example, suppose we want to
%       code x=[.807;-.677] as a binary string with 10 bits representing
%       0.807 and 5 bits representing -.677, where 0<= x(1) <= 3 and
%       -1<= x(2) <= 1.  Then SGA uses the representation:
%
%              0 0 1 1 0 1 1 1 0 1 0 1 0 1 1
%              |-------x(1)-------|--x(2)--|
%
%       Popsize is the number of strings (or parameter vectors) at any
%       generation (or iteration), where generation = 1,..,maxgen. 
%
%       We start out by randomly generating a population of x vectors.
%       Thus, we sample over the entire (bounded) parameter space.
%       For each x in the population, we compute f(x).  We then assign
%       a strength or fitness value to each x.  This value is given
%       by h(x)/sum_{y} h(y) where h(x) = a+b*f(x) for a,b that ensure
%       h(x)>0.  The population is then ``reproduced'' according to
%       fitness as follows:  we spin a roulette wheel that assigns
%       the probability of choosing x in accordance to x's fitness 
%       value; we spin popsize times.  With the new reproduced popu-
%       lation, we assign labels of ``female'' to half the strings
%       and ``male'' to the other half.  With these assignments, we
%       randomly match couples who become the parents of the next 
%       generation.  The two offspring of any couple are found by
%       crossing the genes of the parents and mutating genes.  The
%       probability that crossover and mutation occurs is pcrossover
%       and pmutation, respectively.  If crossover does occur, then
%       the first offspring takes the genes 1 to xsite of the first
%       parent and the genes xsite+1 to sum(lparam) of the second 
%       parent, where xsite is a random number between 1 and sum(lparam)  
%       -1.  The second offspring takes genes 1 to xsite of the second 
%       parent and genes xsite+1 to sum(lparam) of the first.  For 
%       example, if the following two parents:
%
%              0 1 1 0 0 1 0 1 0 1 0 0 1 1 1       -- parent 1
%              1 0 1 1 0 0 1 1 0 1 1 0 0 0 1       -- parent 2
%
%       are matched and crossover is to occur, then a random number
%       (xsite) between 1 and 15 is chosen.  Suppose xsite=4.  Then,
%       the offspring are:
%
%              0 1 1 0 0 0 1 1 0 1 1 0 0 0 1       -- offspring 1
%              1 0 1 1 0 1 0 1 0 1 0 0 1 1 1       -- offspring 2
%
%       Mutation might also occur and would imply that one or more
%       genes in an offspring are flipped from 0 to 1 or vice versa.
%


%--------------------------------------------------------------------------
%                        USER-DEFINED PARAMETERS
%--------------------------------------------------------------------------

%NOTE: To set the default values, set the corresponding element of the 
%      vector default to 1.  Otherwise, specify any subset.  To make sure
%      all of your specifications are chosen, set default=zeros(1,15).


nparam=length(lb);         % number of elements in any x -- make sure that
% no default               %   this is consistent with input to func 


lparam=10*ones(nparam,1);  % lparam(i)=length of binary string representing
% (1)                      %   the ith parameter (default: 10*ones(nparam,1))

popsize=50;                % number of parameter vectors (x) in population
% (2)                      %   (default: 100)

maxgen=50;                 % iteration (or generation) limit (default: 50)
% (3)

pcrossover=.6;             % probability of crossover (in [0,1]) (default:
% (4)                      %   0.6)

pmutation=.01;             % probability of mutation  (in [0,1]) (default:
% (5)                      %   0.01)

printrep=1;                % if printrep=1, print a report every nprint 
nprint=50;                 %   iterations  (0 otherwise)  (defaults: 1,50)
% (6),(7)

reportx=zeros(nparam,1);   %   reportx(i)= number of significant digits
reportx(2)=1;              %     reported for the ith parameter (default:
% (8)                      %     2*ones(nparam,1))

reportf=3;                 %   reportf   = number of significant digits
% (9)                      %     reported for f(x)  (default: 2)

ncol=80;                   % ncol= number of columns to be used for 
% (10)                     %   printing reports  (default: 80)

gradchk=1;                 % gradchk=1 if gradient computed (0 otherwise)
% (11)                     %   (default: 1)

gtol=1e-5;                 % if norm(g)/(1+f(x)) < gtol, code=1 (default:
% (12)                     %   1e-5)

agflag=1;                  % agflag =1 if analytic gradients provided
% (13)                     %   (i.e. [f,g]=func(x)) (0 otherwise) 
                           %   (default: 0)

side=2;                    % side = 1 if 1-sided derivatives (2 if 2-sided)
% (14)                     %   (default: 1)

typsiz=ones(nparam,1);     % typical size of x vector  (default: 
% (15)                     %   ones(nparam,1)

default=[0  0  0  0  0  0  0  0  0   0   0   0   0   0   0];
%        1  2  3  4  5  6  7  8  9  10  11  12  13  14  15 : # of 
%                                                            specification

%--------------------------------------------------------------------------
%                            INITIALIZATIONS
%--------------------------------------------------------------------------

%
%    (a) set defaults if options not chosen or if something set incorrectly
%    

if (default(1) | length(lparam)< nparam); lparam=10*ones(nparam,1); end;
if (default(2) | popsize< 1); popsize=50; end;
if (default(3) | maxgen < 1); maxgen=1000; end;
if (default(4) | pcrossover< 0 | pcrossover>1); pcrossover=.6; end;
if (default(5) | pmutation < 0 | pmutation >1); pmutation=.01; end;
if default(6); printrep=1; end;
if default(7); nprint=50; end;
if (default(8) | length(reportx)< nparam); reportx=2*ones(nparam,1); end;
if default(9); reportf=2; end;
if default(10); ncol=80; end;
if default(11); gradchk=1; end;
if default(12); gtol=1e-5; end;
if default(13); agflag=0; end;
if default(14); side=1; end;
if default(15); typsiz=ones(nparam,1); end;
if min(typsiz)<0; typsiz=abs(typsiz); end;

%
%    (b) initialize data 
%

format short e
code=0;
if rem(popsize,2)~=0; 
  error('Popsize must be even-numbered')
end;
gen=0;
nmutation=0;
ncross=0;
fmultiple=2;
typsiz=abs(typsiz(:));

%
%    (c) inititialize a population at random
%

lchrom=sum(lparam);
tem=length(side);
if tem<nparam;
  side=side(1)*ones(nparam,1);
end;
oldpopc=rand(popsize,lchrom)<.5;                         
oldpopx=decode(oldpopc,nparam,lparam,ub,lb);    
for j=1:popsize;
  eval(['oldpopo(j,1)=',func,'(oldpopx(j,:));'])
end;
oldpopp1=zeros(popsize,1);
oldpopp2=oldpopp1;
oldpopxs=oldpopp1;

%
%    (d) initial statistics
%

[maxf,minf,avgf,sumfitness,best]=statistics(oldpopo,popsize);
xbest=oldpopx(best,:);
fbest=oldpopo(best);
if gradchk;
  if agflag;
    eval(['[tem1,gbest]=',func,'(xbest);'])
  else;
    dev=diag(sqrt(1e-15)*max(abs(xbest'),typsiz));   
    sdev=diag(side-1)*dev;
    for i=1:nparam;
      eval(['gbest(i,1)=(',func,'(xbest+dev(i,:))-',func,  ..
            '(xbest-sdev(i,:)))/(side(i)*dev(i,i));'])
    end;
  end;
  if (norm(gbest) < gtol*(1+abs(fbest)) ); 
    code=1;
    save sgadata
    return
  end;
end;

if minf<0;
  [oldpopf,sumfitness]=scalepop(oldpopo,maxf,minf,avgf,fmultiple);
else;
  oldpopf=oldpopo;
end;

%
%    (e) initial report
%

disp(' ')
disp('--------------------------------------------------------------------------------')
disp('                A SIMPLE GENETIC ALGORITHM FOR OPTIMIZATION ')
disp('--------------------------------------------------------------------------------')
disp(' ')
disp(' ')
disp('  SGA Parameters')
disp('  --------------')
disp(' ')
fprintf('  Population Size (popsize)          = %g\n',popsize)
fprintf('  Number of Elements in x (nparam)   = %g\n',nparam)
for j=1:nparam;
  fprintf('    Chromosome Length, Max and Min Values for Parameter %g: ',j) 
  fprintf('%g, %g, %g\n',lparam(j),ub(j),lb(j))
end;
fprintf('  Chromosome Length (lchrom)         = %g\n',lchrom)
fprintf('  Maximum Generations (maxgen)       = %g\n',maxgen)
fprintf('  Crossover Probability (pcrossover) = %g\n',pcrossover)
fprintf('  Mutation Probability (pmutation)   = %g\n',pmutation)
disp(' ')
disp(' ')
disp('  Initial Generation Statistics')
disp('  -----------------------------')
disp(' ')
fprintf('  Initial population maximum fitness = %g\n',maxf)
fprintf('  Initial population minimum fitness = %g\n',minf)
fprintf('  Initial population average fitness = %g\n',avgf)
fprintf('  Initial population sum of fitness  = %g\n',sumfitness)
disp(' ')
disp(' ')
%
%    (f) initialize strings for printing reports
%
if printrep;
  l1=length(int2str(popsize))+1;
  l2=max(lchrom,6);
  l3=max(7,2*(l1-1)+3);
  l4=max(7,length(int2str(lchrom))+1);
  l5=sum(reportx+7*(reportx>0));
  xon=find(reportx>0);
  l6=reportf+7*(reportf>0);
  l7=length(int2str(maxgen));
  fhalf=l1+l2+l5+l6+1;
  shalf=l3+l4+l2+l5+l6+2;
  ltot=fhalf+shalf;
  blanks=setstr(32*ones(1,ltot));
  str=[blanks(1:ceil(l2/2-3)),'string',blanks(1:floor(l2/2-3))];
  if l5;
    str=[str,blanks(1:ceil((l5-1)/2)),'x',blanks(1:floor((l5-1)/2))];
  end;
  if l6;
    str=[str,blanks(1:ceil((l6-7)/2)),'fitness',blanks(1:floor((l6-7)/2))];
  end;
  str=['#',blanks(1:l1-1),str,'   ',blanks(1:ceil((l3-7)/2)),'parents', ..
       blanks(1:floor((l3-7)/2)+ceil((l4-7)/2)),' xsite ', ..
       blanks(1:floor((l4-7)/2)),str];
  tem1=(fhalf-11-l7)/2; tem2=(shalf-11-l7)/2;
  genspot=[ceil(tem1)+11;2*tem1+22+l7+ceil(tem2)];
  str=[blanks(1:ceil(tem1)),'Generation ',blanks(1:floor(tem1)+l7+  ..
       ceil(tem2)),'Generation ',blanks(1:floor(tem2)+l7);
       blanks(1:ltot);str;setstr(45*ones(1:ltot))];
  for i=1:popsize;
    block1(i,:)=[sprintf(['%',int2str(l1-1),'g'],i),' '];
  end;
  if lchrom<6;
    block1=[block1,setstr(32*ones(popsize,lchrom-6))];
  end;
  block2=setstr(ones(popsize,1)*[32,124,32,40]);
  tem1=max(l4,4);
  block3=setstr(ones(popsize,1)*[41,32*ones(1,l4-2-length(int2str(lchrom)))]);
  block4=setstr(ones(popsize,1)*[32*ones(1,2)]);
  if lchrom<6;
    block4=[block4,setstr(ones(popsize,1)*[32*ones(1,6-lchrom)])];
  end;
  line=setstr(45*ones(1,ncol));
  if ltot>ncol;
    box2=[line;
          setstr(32*ones(1,ceil((ncol-25)/2))),'Population Report (cont.)', ..
          setstr(32*ones(1,floor((ncol-25)/2)))];
    block5=[setstr(32*ones(3,ncol-rem(ltot-ncol,ncol-l1)));
            setstr(45*ones(1,ncol-rem(ltot-ncol,ncol-l1)));
            setstr(32*ones(popsize,ncol-rem(ltot-ncol,ncol-l1)))];
    nbox=fix((ltot-ncol)/(ncol-l1))+1;
  else;
    block5=[setstr(32*ones(3,ncol-ltot));setstr(45*ones(1,ncol-ltot));
            setstr(32*ones(popsize,ncol-ltot))];
    nbox=0;
  end;
  box1=[line;
        setstr(32*ones(1,ceil((ncol-17)/2))),'Population Report', ..
        setstr(32*ones(1,floor((ncol-17)/2)))];
end;

%--------------------------------------------------------------------------
%                               MAIN LOOP
%--------------------------------------------------------------------------

while gen < maxgen;
  gen=gen+1;
  j=1;
  while j<popsize;
    %  (a) spin roulette wheel to select 
    mate1=select(popsize,sumfitness,oldpopf);
    mate2=select(popsize,sumfitness,oldpopf);
    if rand<pcrossover;
      jcross=1+floor((lchrom-1)*rand);
      ncross=ncross+1;
    else;
      jcross=lchrom;
    end;
    newpopc(j,:)=[abs(oldpopc(mate1,1:jcross)-(rand(1,jcross)<pmutation)), ..
        abs(oldpopc(mate2,jcross+1:lchrom)-(rand(1,lchrom-jcross)<pmutation))];
    newpopc(j+1,:)=[abs(oldpopc(mate2,1:jcross)-(rand(1,jcross)<pmutation)), ..
        abs(oldpopc(mate1,jcross+1:lchrom)-(rand(1,lchrom-jcross)<pmutation))];
    newpopx(j:j+1,:)=decode(newpopc(j:j+1,:),nparam,lparam,ub,lb);
    eval(['newpopo(j,1)=',func,'(newpopx(j,:));'])
    eval(['newpopo(j+1,1)=',func,'(newpopx(j+1,:));'])
    newpopp1(j:j+1,1)=mate1*[1;1];
    newpopp2(j:j+1,1)=mate2*[1;1];
    newpopxs(j:j+1,1)=jcross*[1;1];
    j=j+2;
  end;
  [maxf,minf,avgf,sumfitness,best]=statistics(newpopo,popsize);
  if newpopo(best)>fbest; 
    xbest=newpopx(best,:); 
    fbest=newpopo(best);
    fprintf(' Best objective function at generation %g: %g\n',gen,fbest)
    if gradchk;
      if agflag;
        eval(['[fbest,gbest]=',func,'(xbest);'])
      else;
        dev=diag(sqrt(1e-15)*max(abs(xbest'),typsiz));   
        sdev=diag(side-1)*dev;
        for i=1:nparam;
          eval(['gbest(i,1)=(',func,'(xbest+dev(i,:))-',func,  ..
                '(xbest-sdev(i,:)))/(side(i)*dev(i,i));'])
        end;
      end;
      disp(' Associated parameter and gradient vectors:')
      disp([xbest',gbest])
      if (norm(gbest) < gtol*(1+abs(fbest)) ); 
        code=1;
        save sgadata
        return
      end;
    else;
      disp(' Associated parameter vector:')
      disp(xbest')
   end;
  end;
  if minf<0;
    [newpopf,sumfitness]=scalepop(newpopo,maxf,minf,avgf,fmultiple);
  else;
    newpopf=newpopo;
  end;
  if ~rem(gen,nprint);
    xo=[ ]; xn=[ ]; 
    fo=[ ]; fn=[ ];
    for i=1:popsize;
      tem2=[ ]; tem3=[ ]; 
      for j=1:length(xon);
        tem1=['%',int2str(reportx(xon(j))+7),'.',  ..
              int2str(reportx(xon(j))-1),'e'];
        tem2=[tem2,sprintf(tem1,oldpopx(i,xon(j)))];
        tem3=[tem3,sprintf(tem1,newpopx(i,xon(j)))];
      end;
      if l5;
        xo(i,:)=tem2;
        xn(i,:)=tem3;
      end;
      if l6;
        tem1=['%',int2str(reportf+7),'.',int2str(reportf-1),'e'];
        fo(i,:)=sprintf(tem1,oldpopo(i,1));
        fn(i,:)=sprintf(tem1,newpopo(i,1));
      end;
      tem1=['%',int2str(l1-1),'g'];
      p1(i,:)=sprintf(tem1,newpopp1(i));
      p2(i,:)=sprintf(tem1,newpopp2(i));
      tem1=['%',int2str(length(int2str(lchrom))),'g'];
      xs(i,:)=sprintf(tem1,newpopxs(i));
    end;
    tem1=['%',int2str(l7),'g'];
    str(1,genspot(1)+1:genspot(1)+l7)=sprintf(tem1,gen-1);
    str(1,genspot(2)+1:genspot(2)+l7)=sprintf(tem1,gen);
    disp(' ')
    box=[[str;block1,setstr(oldpopc+48),xo,fo,block2,p1, ..
          setstr(32*ones(popsize,1)),p2,block3,xs,block4, ..
          setstr(newpopc+48),xn,fn],block5];
    disp([box1;box(:,1:ncol);line])
    disp(' ')
    for i=1:nbox;
      disp([box2;box(:,1:l1),box(:,ncol+(i-1)*(ncol-l1)+1:ncol+i*(ncol-l1)); ..
            line])
    end;
    fprintf('  Statistics Report for Generation %g:\n',gen)
    fprintf('    maxf  = %12.4e, minf  = %12.4e, avgf  = %12.4e\n',  ..
                 maxf,minf,avgf)
    fprintf('    fbest = %12.4e \n',fbest)
    if gradchk;
      disp('    xbest =      gbest =')
      disp([xbest',gbest])
    else;
      disp('    xbest = ')
      disp(xbest')
    end;
    disp(line)
    disp(' ')
  end;
  oldpopc=newpopc;
  oldpopx=newpopx;
  oldpopf=newpopf;
  oldpopo=newpopo;
  oldpopp1=newpopp1;
  oldpopp2=newpopp2;
end;
code=2;
save sgadata
