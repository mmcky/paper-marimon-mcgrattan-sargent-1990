function [CS,last]=ga(CS,nselect,pcross,pmutation,crowdingfactor, ..
                         crowdingsubpop,M,l2,smultiple,last,type) 
lchrom=l2+2;
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
for j=1:nselect;
  mate1=select(M,sumstrength,CS(:,lchrom+1));
  mate2=select(M,sumstrength,CS(:,lchrom+1));

  % Crossover 

  if rand<pcross;
    jcross=1+floor((l2-1)*rand);
    ncross=ncross+1;
  else;
    jcross=l2;
  end;
  rnd1=(rand(1,l2)<pmutation);
  rnd2=(rand(1,2)<pmutation);
  v=[CS(mate1,1:jcross),CS(mate2,jcross+1:l2)];
  av=(CS(mate1,l2+3)+CS(mate2,l2+3))*.5;
  child1=[(1-rnd1).*v+rnd1.*(rem(v+ceil(rand(v)*2)+1,3)-1), ..
          abs(CS(mate1,l2+1:lchrom)-rnd2),av];

  rnd3=(rand(1,l2)<pmutation);
  rnd4=(rand(1,2)<pmutation);
  v=[CS(mate2,1:jcross),CS(mate1,jcross+1:l2)];
  child2=[(1-rnd3).*v+rnd3.*(rem(v+ceil(rand(v)*2)+1,3)-1), ..
          abs(CS(mate2,l2+1:lchrom)-rnd4),av];
  nmutation=nmutation+sum([rnd1,rnd2,rnd3,rnd4]);

  % Crowding
  mort1=crowding(child1,CS,crowdingfactor,crowdingsubpop,M,l2); 
  tem=find(~(last(:,type)-mort1));
  if ~isempty(tem); last(tem,type)=zeros(length(tem),1); end;
  sumstrength=sumstrength-CS(mort1,l2+3)+av;
  CS(mort1,:)=[child1,0,0,iteration];

  mort2=crowding(child2,CS,crowdingfactor,crowdingsubpop,M,l2);
  tem=find(~(last(:,type)-mort2));
  if ~isempty(tem); last(tem,type)=zeros(length(tem),1); end; 
  sumstrength=sumstrength-CS(mort2,l2+3)+av;
  CS(mort2,:)=[child2,0,0,iteration];
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
