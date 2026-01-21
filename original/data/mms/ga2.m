function [CS,last]=ga2(CS,nselect,pcross,pmutation,crowdingfactor, ..
          crowdingsubpop,M,l2,smultiple,last,type,iteration,itpos,propused) 
lchrom=l2+1;
[maxs,mins,avgs,sumstrength]=statistics(CS(:,lchrom+1),M);
if mins<0;
%  [a,b]=scalestr(maxs,mins,avgs,smultiple);
%  CS(:,lchrom+1)=a*CS(:,lchrom+1)+b;
  CS(:,lchrom+1)=CS(:,lchrom+1)-mins;
  sumstrength=sum(CS(:,lchrom+1));
end;
disp('Pair    Mate1   Mate2   SiteCross   Mort1   Mort2')
disp('-------------------------------------------------')
ncross=0; nmutation=0;
ncalled=propused*M;

for j=1:nselect;
  tem1=CS(:,itpos-1)+1;
  sumuse=sum(CS(:,itpos-1)+1);
  ind=[ ];
  for k=1:ncalled;
    index=select(M,sumuse,tem1);
    ind=[ind;index];
    sumuse=sumuse-tem1(index);
    tem1(index)=0;
  end;
  lind=length(ind);
  tem1=CS(ind,lchrom+1)+1;
  tem2=sum(tem1);
  mate1=select(lind,tem2,tem1);
  mate2=select(lind,tem2,tem1);

  mate1=ind(mate1);
  mate2=ind(mate2);

  % Crossover 

  if rand<pcross;
    jcross=1+floor((l2-1)*rand);
    ncross=ncross+1;
  else;
    jcross=l2;
  end;
  rnd1=(rand(1,l2)<pmutation);
  rnd2=(rand<pmutation);
  v=[CS(mate1,1:jcross),CS(mate2,jcross+1:l2)];
  av=(CS(mate1,l2+2)+CS(mate2,l2+2))*.5;
  child1=[(1-rnd1).*v+rnd1.*(rem(v+ceil(rand(v)*2)+1,3)-1), ..
          abs(CS(mate1,l2+1)-rnd2),av];

  rnd3=(rand(1,l2)<pmutation);
  rnd4=(rand<pmutation);
  v=[CS(mate2,1:jcross),CS(mate1,jcross+1:l2)];
  child2=[(1-rnd3).*v+rnd3.*(rem(v+ceil(rand(v)*2)+1,3)-1), ..
          abs(CS(mate2,l2+1)-rnd4),av];
  nmutation=nmutation+sum([rnd1,rnd2,rnd3,rnd4]);

  % Crowding
  mort1=crowding2(child1,CS,crowdingfactor,crowdingsubpop,M,l2);
  tem1=find(~(last(:,type)-mort1));
  if ~isempty(tem1); last(tem1,type)=zeros(length(tem1),1); end;
  sumstrength=sumstrength-CS(mort1,l2+2)+av;
  CS(mort1,:)=[child1,zeros(1,itpos-l2-3),iteration];

  mort2=crowding2(child2,CS,crowdingfactor,crowdingsubpop,M,l2);
  tem1=find(~(last(:,type)-mort2));
  if ~isempty(tem1); last(tem1,type)=zeros(length(tem1),1); end;
  sumstrength=sumstrength-CS(mort2,l2+2)+av;
  CS(mort2,:)=[child2,zeros(1,itpos-l2-3),iteration];
  disp([j,mate1,mate2,jcross,mort1,mort2])
end;
disp(' ')
disp('Statistics Report')
disp('-----------------')
disp(' ')
fprintf(' Average strength    = %g\n',avgs)
fprintf(' Maximum strength    = %g\n',maxs)
fprintf(' Minimum strength    = %g\n',mins)
fprintf(' Sum of strength     = %g\n',sumstrength)
fprintf(' Number of crossings = %g\n',ncross)
fprintf(' Number of mutations = %g\n',nmutation)
disp(' ')
if mins<0;
%  CS(:,lchrom+1)=(CS(:,lchrom+1)-b)/a;
  CS(:,lchrom+1)=CS(:,lchrom+1)+mins;
end;
