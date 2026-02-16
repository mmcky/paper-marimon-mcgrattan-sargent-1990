function [CS,last]=ga3(CS,ctype,nselect,pcross,pmutation,crowdingfactor, ..
            crowdingsubpop,nclass,lcond,last,iteration,propused,uratio)
actpos=lcond+1;                         
fitpos=lcond+2;               %  Bit position:             Information:
if ctype==1;                  %  -------------             ------------
  usepos=lcond+4;             %    actpos                  action
  itpos=lcond+5;              %    fitpos                  strength
else;                         %    usepos                  no. times rule used
  usepos=lcond+3;             %    itpos                   iteration when rule
  itpos=lcond+4;              %                            originated
end;
tem1=max(CS(:,usepos));
cankill=find( CS(:,fitpos)<uratio(1) | CS(:,usepos)/(1+tem1(1)) <uratio(2));
if isempty(cankill);          %  cankill is a vector of indices of CS marking
  return;                     %    rules that are candidates for genetic
end;                          %    elimination
numkill=length(cankill);
if 2*nselect>numkill;         %  if too many pairs have been selected, 
  nselect=(numkill+rem(numkill,2))/2; %    change nselect
end;
[maxb,minb,avgb]=statistics(CS(:,fitpos),nclass);
if minb<0;                    %  scale classifier if any strengths are <0
  CS(:,fitpos)=CS(:,fitpos)-minb;
end;
ncross=0; nmutation=0;
ncalled=propused*nclass;
disp('Pair    Mate1   Mate2  Cross Points(2)  In   Mort1   Mort2  Iteration')
disp('---------------------------------------------------------------------')
for j=1:nselect;
  if ncalled<nclass;
    tem1=CS(:,usepos)+1;       
    sumuse=sum(CS(:,usepos)+1);
    ind=[ ];
    for k=1:ncalled;          %  a subset with the most used rules will
      index=select(nclass,sumuse,tem1);  % be used for choosing mates
      ind=[ind;index];
      sumuse=sumuse-tem1(index);
      tem1(index)=0;
    end;
  else;
    ind=[1:nclass]';
  end;
  lind=length(ind);           
  tem1=CS(ind,fitpos)+1;
  tem2=sum(tem1);             %  spin a roulette wheel to select 2 high
  mate1=select(lind,tem2,tem1); %  strength mates
  mate2=select(lind,tem2,tem1);

  mate1=ind(mate1);
  mate2=ind(mate2);

  av=(CS(mate1,fitpos)+CS(mate2,fitpos))*.5;
  child1=[CS(mate1,1:lcond+1),av];
  child2=[CS(mate2,1:lcond+1),av];

  % Crossover 

  jcross=sort(1+floor((lcond+1)*rand(2,1)));
  in=rand>.5;
  if in;
    ind=[jcross(1):jcross(2)-1];
  else;
    ind=[1:jcross(1)-1,jcross(2):lcond];
  end;
  if ~isempty(ind);
    tem1=child1(ind);
    tem2=child2(ind);
    tem3= abs( (tem1<0).*tem2+(tem1>=0).*tem1 - (tem2<0).*tem1-  ..
              (tem2>=0).*tem2 );
    child1(ind)=tem1.*(~tem3)-tem3;
    child2(ind)=tem2.*(~tem3)-tem3;
  end;
  if ~(in & jcross(1)~=jcross(2)) | (~in & jcross(2)-jcross(1)==lcond);
    ncross=ncross+1;
  end;

  % Crowding

  tem1=min(nclass*crowdingsubpop/numkill,1);
  mort1=crowdin3(child1,CS(cankill,:),crowdingfactor,tem1,numkill,lcond);
  mort1=cankill(mort1);              %  find the first string to be       
  cankill=cankill(cankill~=mort1);   %  killed (one in the cankill vector)
  numkill=numkill-1;
  if ctype==2;
    tem1=find(~(last-mort1));
    if ~isempty(tem1); last(tem1)=zeros(length(tem1),1); end;
  end;
  if ctype==1;
    CS(mort1,:)=[child1,CS(mate1,fitpos+1:usepos),iteration];
  else;
    avu=CS(mate1,usepos);
    CS(mort1,:)=[child1,avu,iteration];
  end;

  if isempty(cankill);
    mort2=0;                         %  if cankill is not empty, find
  else;                              %  the second string to be killed
    tem1=min(nclass*crowdingsubpop/numkill,1);
    mort2=crowdin3(child2,CS(cankill,:),crowdingfactor,tem1,numkill,lcond);
    mort2=cankill(mort2);
    cankill=cankill(cankill~=mort2);
    numkill=numkill-1;
    if ctype==2;
      tem1=find(~(last-mort2));
      if ~isempty(tem1); last(tem1)=zeros(length(tem1),1); end;
    end;
    sumstrength=sumstrength-CS(mort2,fitpos)+av;
    if ctype==1;
      CS(mort2,:)=[child2,CS(mate2,fitpos+1:usepos),iteration];
    else;
      avu=.5*(avu+CS(mate2,usepos));
      CS(mort2,:)=[child2,avu,iteration];
      CS(mort1,usepos)=avu;
    end;
  end;
  disp([j,mate1,mate2,jcross(1),jcross(2),in,mort1,mort2,iteration])
end;
if minb<0;                    % rescale classifier system's strengths
  CS(:,fitpos)=CS(:,fitpos)+minb;
end;
[maxa,mina,avga]=statistics(CS(:,fitpos),nclass);
disp(' ')                     % display results:
fprintf('Statistics Report at iteration %g\n',iteration) 
disp('-----------------------------------')
disp(' ')
fprintf(' Average strength before genetics   = %g\n',avgb)
fprintf(' Maximum strength before genetics   = %g\n',maxb)
fprintf(' Minimum strength before genetics   = %g\n',minb)
fprintf(' Number of crossings = %g\n',ncross)
fprintf(' Number of mutations = %g\n',nmutation)
fprintf(' Average strength after genetics   = %g\n',avga)
fprintf(' Maximum strength after genetics   = %g\n',maxa)
fprintf(' Minimum strength after genetics   = %g\n',mina)
disp(' ')
