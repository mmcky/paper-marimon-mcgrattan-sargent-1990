%  A Simple Classifier System applied to Wicksell N-tangles
%  --------------------------------------------------------
%
%
%  Initializations:
%
%
%   (a) initialize parameters
%
winitial 
total=nagents*ntypes;
if rem(total,2)>0; 
  error('The total population of the economy must be an even number');
end;
[row,l]=size(bnames);
l2=2*l;
nselect=round(proportionselect*nclassifiers*.5);
smultiple=2;
%
%   (b) initialize the matrices with returns R<i>, the classifier 
%       systems CS<i>, the meeting<i> matrices, the tottrade<i>
%       matrices, the conswtrade<i> matrices, the conswnotrade<i>
%       matrices, the transition<i> matrices, i=1,...,ntypes, and 
%       the holding matrix
%   
%
o=ones(ntypes,l);
mo=-o;
z=zeros(ntypes,l);
e=eye(ntypes);
e2=eye(2*ntypes);
tem1=[    mo mo mo(:,1) mo(:,1) bnames     mo  z(:,1) z(:,1);
          mo mo mo(:,1) mo(:,1) bnames     mo  z(:,1) o(:,1);
      bnames mo  o(:,1) mo(:,1)     mo bnames  o(:,1) z(:,1);
      bnames mo  o(:,1) mo(:,1)     mo bnames  o(:,1) o(:,1);
          mo mo  z(:,1) mo(:,1) bnames     mo mo(:,1) z(:,1);
          mo mo  z(:,1) mo(:,1) bnames     mo mo(:,1) o(:,1)];
prob(3,:)=1-sum(prob);
cs=cumsum(prob);
for i=1:ntypes;
  k=int2str(i);
  eval(['R',k,'=[tem1,[e2;e2;e2]*[bnames -storecosts(i,:)'';', ..
        'ones(ntypes,1)*bnames(produces(i),:) -ones(ntypes,1)*', ..
        '(storecosts(i,produces(i))+prodcosts(i))+e(:,i)*utility(i)]];'])
  tem2=[ ];
  for j=1:nclassifiers;
    tem2=[tem2;sum(ones(3,1)*rand(1,l2)-cs>0)-1];
  end;
  eval(['CS',k,'=[tem2,round(rand(nclassifiers,2)),strength(:,i),', ..
        'zeros(nclassifiers,3)];'])
  eval(['meeting',k,'=zeros(ntypes);'])
  eval(['tottrade',k,'=zeros(ntypes);'])
  eval(['conswtrade',k,'=zeros(ntypes);'])
  eval(['conswnotrade',k,'=zeros(ntypes);'])
  eval(['transition',k,'=zeros(ntypes);'])
end;
holding=zeros(ntypes);
%disp('TWO OF THE CLASSIFIERS HAVE HAD ADJUSTMENTS MADE')
%CS2(1:2,:)=[-1 1 -1 -1 1 1 300 0 0 0;
%            -1 1 -1 -1 1 0 300 0 0 0];
%CS3(1,:)=[-1 -1 -1 1 1 1 300 0 0 0];

%
%  (c) initialize the storages for the population
%
popstorage=bnames(ceil(rand(total,1)*ntypes),:);
%
%  (d) initialize the indices for winning classifiers of the iteration before
%      who will be rewarded bid payments
%
last=zeros(nagents,ntypes);
%
%  (e) and print out parameters and original classifier systems.
%
disp(' ')
disp('Parameter Specificatons')
disp('-----------------------')
disp(' ')
fprintf(' Number of agent types = %g\n',ntypes)
fprintf(' Number of agents of a particular type = %g\n',nagents)
fprintf(' Number of strings in classifier systems = %g\n',nclassifiers)
fprintf(' Number of periods between displays of classifiers = %g\n',dispclass)
fprintf(' Number of periods between genetic algorithm calls = %g\n',Tga)
s=' Proportion of strings in classifier chosen for reproduction = %g\n';
fprintf(s,proportionselect)
fprintf(' Probability of crossover = %g\n',pcross)
fprintf(' Probability of mutation = %g\n',pmutation)
fprintf(' Crowding subpopulation = %g\n',crowdingsubpop)
fprintf(' Crowding factor = %g\n',crowdingfactor)
disp(' Good labels:')
disp(bnames)
disp(' Good types produced:')
disp(produces)
disp(' Costs of storing goods:')
disp(storecosts)
disp(' Costs of producing goods:')
disp(prodcosts)
disp(' Utility: ')
disp(utility)
disp(' Bid1:')
disp(bid1)
disp(' Bid2:')
disp(bid2)
disp(' Taxes:')
disp(tax)
disp(' Probabilities of -1,0 in generating classifier strings:')
disp(prob)
disp(' ')
disp(' ')
disp(' ')
disp(' ')
disp('Initial Classifier Systems')
disp('--------------------------')
disp(' ')
for i=1:ntypes;
  fprintf('  Classifier System for Type %g Agents: \n',i)
  disp('  -------------------------------------')
  disp(' ')
  eval(['disp([[1:nclassifiers]'',CS',int2str(i),',[1:nclassifiers]''])'])
  disp(' ')
end;
%
%
list=[ ];
for i=1:ntypes;
  list=[list; [1:nagents]',ones(nagents,1)*i];
end;
%
%  For maxit iterations, 
%
for it=1:maxit
  for i=1:ntypes;
    eval(['Tax(:,i)=tax(i)*abs(CS',int2str(i),'(:,l2+3));'])
  end;
  flag=0;
  %
  %  randomly match agents and ..
  %
  tem1=list;
  for i=1:total;
    pos=1+floor(rand*(total-i+1));
    mate1(i,:)=tem1(pos,:);
    tem1=tem1([1:pos-1,pos+1:total-i+1],:);
  end;
  halftot=round(.5*total);
  mate2=mate1(halftot+1:total,:);
  mate1=mate1(1:halftot,:);
  %
  %  for each pair of mates i,j, where i,j=1,2,...1/2*total: 
  %
  for i=1:halftot;
    %
    % (a) get conditions: [own storage, match's storage],
    %
    condition1=[popstorage(mate1(i,1)+nagents*(mate1(i,2)-1),:), ..
                popstorage(mate2(i,1)+nagents*(mate2(i,2)-1),:)];
    condition2=condition1([l+1:l2,1:l]);
    %
    % (b) get strings type1 and type2 giving agent types,
    %
    type1=int2str(mate1(i,2));
    type2=int2str(mate2(i,2));
    %
    % (c) update meeting<i> and holding matrices, where 
    %     meeting<i>(j,k) =# of times a type-i agent carrying good j 
    %                      met an agent carrying good k
    %     holding(i,j)    =# of times a type-i agent is holding a good
    %                      of type j
    %
    ind1=find(~sum(abs( (bnames-ones(ntypes,1)*condition1(1:l))' )));
    ind2=find(~sum(abs( (bnames-ones(ntypes,1)*condition2(1:l))' )));
    eval(['meeting',type1,'(ind1,ind2)=meeting',type1,'(ind1,ind2)+1;'])
    eval(['meeting',type2,'(ind2,ind1)=meeting',type2,'(ind2,ind1)+1;'])
    holding(mate1(i,2),ind1)=holding(mate1(i,2),ind1)+1;
    holding(mate2(i,2),ind2)=holding(mate2(i,2),ind2)+1;
    %
    % (d) find indices of classifiers in CS matching conditions and
    %     if there are no matches, replace a string with the condition,
    %
    cstr=['CS',type1,'(:,1:l2)'];
    eval(['ind3=find(~sum(abs(( (',cstr,'>=0).*',cstr,'+(',cstr,'<0).*(ones', ..
      '(nclassifiers,1)*condition1)-ones(nclassifiers,1)*condition1)'')))'';'])
    if isempty(ind3); 
      eval(['[ind3,CS',type1,']=create(CS',type1,',nclassifiers,l2,'  ..
            'condition1);'])
      eval(['CS',type1,'(ind3,l2+6)=it;'])
      tem1=find(~(last(:,mate1(i,2))-ind3));
      if ~isempty(tem1); last(tem1,mate1(i,2))=zeros(length(tem1),1); end;
    end;

    cstr=['CS',type2,'(:,1:l2)'];
    eval(['ind4=find(~sum(abs(( (',cstr,'>=0).*',cstr,'+(',cstr,'<0).*(ones', ..
      '(nclassifiers,1)*condition2)-ones(nclassifiers,1)*condition2)'')))'';'])
    if isempty(ind4);
      if mate1(i,2)==mate2(i,2);
        tem2=ones(nclassifiers,1);
        tem2(ind3,1)=zeros(length(ind3),1);
        eval(['[ind4,tem1]=create(CS',type2,'(tem2,:),sum(tem2),l2,',  ..
              'condition2);'])
        eval(['CS',type2,'(tem2,:)=tem1;'])
        ind4=find(cumsum(tem2)==ind4);
        ind4=ind4(1);
      else;
        eval(['[ind4,CS',type2,']=create(CS',type2,',nclassifiers,l2,'  ..
              'condition2);'])
      end;
      eval(['CS',type2,'(ind4,l2+6)=it;'])
      tem1=find(~(last(:,mate2(i,2))-ind2));
      if ~isempty(tem1); last(tem1,mate2(i,2))=zeros(length(tem1),1); end;
    end;
    %
    % (e) find matching classifiers with winning bids ..
    %

    eval(['c1=[ind3,CS',type1,'(ind3,l2+3)];'])
    eval(['c2=[ind4,CS',type2,'(ind4,l2+3)];'])
    win1=find(c1(:,2)>=max(c1(:,2)));
    win2=find(c2(:,2)>=max(c2(:,2)));
    win1=c1(win1(1+floor(rand*length(win1))),:);
    win2=c2(win2(1+floor(rand*length(win2))),:);
    %
    % (f) and their strings,
    %
    eval(['string1=CS',type1,'(win1(1),1:l2+2);'])
    eval(['string2=CS',type2,'(win2(1),1:l2+2);'])
    sp1=sum(string1(1:l)<0)*isempty(find(~sum(abs(bnames-ones(ntypes,1)* ..
        string1(1:l))')))+sum(string1(l+1:l2)<0)*isempty(find(~sum(abs   ..
        (bnames-ones(ntypes,1)*string1(l+1:l2))')));
    sp1=1/(1+sp1);
    sp2=sum(string2(1:l)<0)*isempty(find(~sum(abs(bnames-ones(ntypes,1)* ..
        string2(1:l))')))+sum(string2(l+1:l2)<0)*isempty(find(~sum(abs   ..
        (bnames-ones(ntypes,1)*string2(l+1:l2))')));
    sp2=1/(1+sp2);
    %
    % (g) find next period's storage and return for each type,
    %

    arg=4*l+4;
    tem1=ones(6*ntypes,1)*[string2,string1];
    eval(['ind5=find(~sum(abs(((R',type1,'(:,1:arg)>=0 & tem1>=0).*R',type1, ..
          '(:,1:arg)+(R',type1,'(:,1:arg)<0 | tem1<0).*tem1-tem1)'')));'])
    eval(['next=R',type1,'(ind5(1),arg+1:arg+l+1);'])
    popstorage(mate1(i,1)+nagents*(mate1(i,2)-1),:)=next(1:l);
    ind7=find(~sum(abs( (bnames-ones(ntypes,1)*next(1:l))' )));
    cbid=bid1(mate1(i,2))+bid2(mate1(i,2))*sp1;
    source=last(mate1(i,1),mate1(i,2));
    if source>0;
      rstr=['CS',type1,'(source,l2+3)'];
      eval([rstr,'=',rstr,'+cbid*CS',type1,'(win1(1),l2+3);'])
    end;
    eval(['CS',type1,'(win1(1),l2+3)=CS',type1,'(win1(1),l2+3)*(1-cbid)', ..
          '+next(l+1);'])
    last(mate1(i,1),mate1(i,2))=win1(1);

    tem1=ones(6*ntypes,1)*[string1,string2];
    eval(['ind6=find(~sum(abs(((R',type2,'(:,1:arg)>=0 & tem1>=0).*R',type2, ..
          '(:,1:arg)+(R',type2,'(:,1:arg)<0 | tem1<0).*tem1-tem1)'')));'])
    eval(['next=R',type2,'(ind6(1),arg+1:arg+l+1);'])
    popstorage(mate2(i,1)+nagents*(mate2(i,2)-1),:)=next(1:l);
    ind8=find(~sum(abs( (bnames-ones(ntypes,1)*next(1:l))' )));
    cbid=bid1(mate2(i,2))+bid2(mate2(i,2))*sp2;
    source=last(mate2(i,1),mate2(i,2));
    if source>0;
      rstr=['CS',type2,'(source,l2+3)'];
      eval([rstr,'=',rstr,'+cbid*CS',type2,'(win2(1),l2+3);'])
    end;
    eval(['CS',type2,'(win2(1),l2+3)=CS',type2,'(win2(1),l2+3)*(1-cbid)', ..
          '+next(l+1);'])
    last(mate2(i,1),mate2(i,2))=win2(1);
   
    %
    % (h) update the transition matrices, where
    %     transition<i>(j,k)=# of times a type-i starts with good j and 
    %                        ends with good k
    %
    
    eval(['transition',type1,'(ind1,ind7)=transition',type1,'(ind1,ind7)+1;'])
    eval(['transition',type2,'(ind2,ind8)=transition',type2,'(ind2,ind8)+1;'])

    %
    % (h) update the number of times the rule was called and the number
    %      of exchanges,
    %
    eval(['trade=(R',type1,'(ind5(1),l2+1) & R',type1,'(ind5(1),4*l+3));'])
    eval(['tem1=(trade~=(R',type2,'(ind6(1),l2+1) & R',type2, ..
          '(ind6(1),4*l+3)) );'])
    if tem1;
      error('Check return matrices for possible mistake')
    end;
    eval(['CS',type1,'(win1(1),l2+4:l2+5)=CS',type1,'(win1(1),l2+4:l2+5)', ..
          '+[trade,1];'])
    eval(['CS',type2,'(win2(1),l2+4:l2+5)=CS',type2,'(win2(1),l2+4:l2+5)', ..
          '+[trade,1];'])

    %
    % (i) update the tottrade<i>, conswtrade<i>, and conswnotrade<i> where
    %     tottrade<i>(j,k)    = # of trades that have occurred when a type-i 
    %                           holding good j meets an agent carrying good k
    %     conswtrade<i>(j,k)  = # of times consumption has occurred after a 
    %                           type-i holding good j meets an agent carrying 
    %                           good k and trades
    %     conswnotrade<i>(j,k)= # of times consumption has occurred after a 
    %                           type-i holding good j meets an agent carrying 
    %                           good k and does not trade
    %
    eval(['tottrade',type1,'(ind1,ind2)=tottrade',type1,'(ind1,ind2)+trade;'])
    eval(['tottrade',type2,'(ind2,ind1)=tottrade',type2,'(ind2,ind1)+trade;'])
    if trade;
      eval(['conswtrade',type1,'(ind1,ind2)=conswtrade',type1, ..
            '(ind1,ind2)+R',type1,'(ind5(1),4*l+4);'])
      eval(['conswtrade',type2,'(ind2,ind1)=conswtrade',type2, ..
            '(ind2,ind1)+R',type2,'(ind6(1),4*l+4);'])
    else;
      eval(['conswnotrade',type1,'(ind1,ind2)=conswnotrade',type1, ..
            '(ind1,ind2)+R',type1,'(ind5(1),4*l+4);'])
      eval(['conswnotrade',type2,'(ind2,ind1)=conswnotrade',type2, ..
            '(ind2,ind1)+R',type2,'(ind6(1),4*l+4);'])
    end;
    %
    % (j) and rescale strengths.
    %

    eval(['[maxs,mins,avgs]=statistics(CS',type1,'(:,l2+3),nclassifiers);'])
    if mins<0;
      [a,b]=scalestr(maxs,mins,avgs,smultiple);
      eval(['CS',type1,'(:,l2+3)=a*CS',type1,'(:,l2+3)+b;'])
    end;
    if mate1(i,2)~=mate2(i,2);
      eval(['[maxs,mins,avgs]=statistics(CS',type2,'(:,l2+3),nclassifiers);'])
      if mins<0;
        [a,b]=scalestr(maxs,mins,avgs,smultiple);
        eval(['CS',type2,'(:,l2+3)=a*CS',type2,'(:,l2+3)+b;'])
      end;
    end;
  end;
  %  
  %  Subtract off taxes computed at the beginning of the iteration
  %
  for i=1:ntypes;
    eval(['CS',int2str(i),'(:,l2+3)=CS',int2str(i),'(:,l2+3)-Tax(:,i);'])
  end;

  %
  %  For every Tga periods, run the genetic algorithm.
  %
  if rem(it,20*Tga)==0 | (it==20 & Tga<maxit);
    for i=1:ntypes;
      disp(' ')
      s='  Classifier System for Type %g Agents before Genetic Algorithm: \n';
      fprintf(s,i)
      disp('  ----------------------------------------------------------------')
      disp(' ')
      eval(['disp([[1:nclassifiers]'',CS',int2str(i),',[1:nclassifiers]''])'])
      disp(' ')
      fprintf('  Meeting Matrix for Type %g Agents: \n',i)
      disp('  --------------------------------------')
      eval(['disp(meeting',int2str(i),')'])
      disp(' ')
      fprintf('meeting%g(i,j)=# times a type-%g agent with good i',i,i)
      fprintf(' meets an agent with good j\n') 
      disp(' ')
      fprintf('  Probability of Trade Matrix for Type %g Agents: \n',i)
      disp('  --------------------------------------------------')
      eval(['disp(tottrade',int2str(i),'./meeting',int2str(i),')'])
      disp(' ')
      fprintf('ptrade%g(i,j)=probability that trade occurs when a type-%g',i,i)
      fprintf(' with good j meets an agent with good k\n')
      disp(' ')
      fprintf('  Probability of Consumption Given Trade Matrix for Type ')
      fprintf('%g Agents: \n',i)
      disp('  ----------------------------------------------------------------')
      eval(['disp(conswtrade',int2str(i),'./tottrade',int2str(i),')'])
      disp(' ')
      fprintf('pcons%g(i,j)=probability that consumption occurs when a',i)
      fprintf(' type-%g with good j meets an agent with good k and trades\n',i)
      disp(' ')
      fprintf('  Probability of Consumption Given No Trade Matrix for Type ')
      fprintf('%g Agents: \n',i)
      disp('  ----------------------------------------------------------------')
      eval(['disp(conswnotrade',int2str(i),'./(meeting',int2str(i), ..
            '-tottrade',int2str(i),'))'])
      disp(' ')
      fprintf('pcons%g(i,j)=probability that consumption occurs when a',i)
      fprintf(' type-%g with good j meets an agent with good k',i)
      fprintf(' and does not trade\n')
    end;
    for i=1:ntypes;
      fprintf('Genetic Algorithm for Classifier System %g \n',i)
      if exp1==1;
        eval(['[CS',int2str(i),',last]=ga2(CS',int2str(i),',nselect,pcross', ..
         ',pmutation,crowdingfactor,crowdingsubpop,nclassifiers,l2', ..
         ',smultiple,last,',int2str(i),',it,propmostused);'])
      else;
        eval(['[CS',int2str(i),',last]=ga(CS',int2str(i),',nselect,pcross', ..
         ',pmutation,crowdingfactor,crowdingsubpop,nclassifiers,l2', ..
         ',smultiple,last,',int2str(i),',it);'])
      end;
    end;  
    for i=1:ntypes;
      disp(' ')
      s='  Classifier System for Type %g Agents after Genetic Algorithm: \n';
      fprintf(s,i)
      disp('  ----------------------------------------------------------------')
      disp(' ')
      eval(['disp([[1:nclassifiers]'',CS',int2str(i),',[1:nclassifiers]''])'])
    end;
    flag=1;
  end;
  %
  %  Print out the results for iteration "it".
  %
  disp(' ')
  disp(' ')
  fprintf('Results for Iteration %g: \n',it)
  disp('-------------------------')
  disp(' ')
  for i=1:ntypes;
    fprintf('  Histogram for Type %g Agents and Transition Matrix:\n',i)
    disp('  -------------------------------------------------------')
%   disp(' ')
    tem1=[ ];
    for j=1:ntypes;
      no=sum(~sum(abs( (popstorage(nagents*(i-1)+1:nagents*i,:)-  ..
         ones(nagents,1)*bnames(j,:))' )));
      tem1=[tem1;no];
%      fprintf('  Number holding good %g: %g \n',j,no);
    end;
    eval([' disp([tem1,transition',int2str(i),  ..
          './(holding(i,:)''*ones(1,ntypes)) ]) '])
    disp(' ')
    if rem(it,dispclass)==0 & flag==0;
      disp(' ')
      fprintf('  Classifier System for Type %g Agents: \n',i)
      disp('  -------------------------------------')
      disp(' ')
      eval(['disp([[1:nclassifiers]'',CS',int2str(i),',[1:nclassifiers]''])'])
      disp(' ')
      fprintf('  Meeting Matrix for Type %g Agents: \n',i)
      disp('  --------------------------------------')
      eval(['disp(meeting',int2str(i),')'])
      disp(' ')
%     fprintf('meeting%g(i,j)=# times a type-%g agent with good i',i,i)
%     fprintf(' meets an agent with good j\n')
%     disp(' ')
      fprintf('  Probability of Trade Matrix for Type %g Agents: \n',i)
      disp('  --------------------------------------------------')
      eval(['disp(tottrade',int2str(i),'./meeting',int2str(i),')'])
      disp(' ')
%     fprintf('ptrade%g(i,j)=probability that trade occurs when a type-%g',i,i)
%     fprintf(' with good j meets an agent with good k\n')
%     disp(' ')
      fprintf('  Probability of Consumption Given Trade Matrix for Type ')
      fprintf('%g Agents: \n',i)
      disp('  ----------------------------------------------------------------')
      eval(['disp(conswtrade',int2str(i),'./tottrade',int2str(i),')'])
      disp(' ')
%     fprintf('pcons%g(i,j)=probability that consumption occurs when a',i)
%     fprintf(' type-%g holding good j meets an agent with good k',i)
%     fprintf(' and trades\n')
%     disp(' ')
      fprintf('  Probability of Consumption Given No Trade Matrix for Type ')
      fprintf('%g Agents: \n',i)
      disp('  ----------------------------------------------------------------')
      eval(['disp(conswnotrade',int2str(i),'./(meeting',int2str(i), ..
            '-tottrade',int2str(i),'))'])
      disp(' ')
%     fprintf('pcons%g(i,j)=probability that consumption occurs when a',i)
%     fprintf(' type-%g with good j meets an agent with good k',i)
%     fprintf(' and does not trade\n')
    end;
  end;
end;
