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
%
%   (b) initialize the matrices with returns Ri, i=1,...,ntypes
%       and the classifier systems CSi, i=1,...,ntypes
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
        'zeros(nclassifiers,2)];'])
end;
%
%  (c) initialize the storages for the population
%
popstorage=bnames(ceil(rand(total,1)*ntypes),:);
%
%  (d) and print out original classifier systems.
%
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
  %
  %  randomly match agents and ..
  %
  tem1=list;
  for i=1:total;
    pos=ceil(rand*(total-i+1));
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
    % (c) find indices of classifiers in CS matching conditions and
    %     if there are no matches, replace a string with the condition,
    %
    cstr=['CS',type1,'(:,1:l2)'];
    eval(['ind1=find(~sum(abs(( (',cstr,'>=0).*',cstr,'+(',cstr,'<0).*(ones', ..
      '(nclassifiers,1)*condition1)-ones(nclassifiers,1)*condition1)'')))'';'])
    if isempty(ind1); 
      eval(['[ind1,CS',type1,']=create(CS',type1,',nclassifiers,l2,'  ..
            'condition1);'])
    end;

    cstr=['CS',type2,'(:,1:l2)'];
    eval(['ind2=find(~sum(abs(( (',cstr,'>=0).*',cstr,'+(',cstr,'<0).*(ones', ..
      '(nclassifiers,1)*condition2)-ones(nclassifiers,1)*condition2)'')))'';'])
    if isempty(ind2);
      if mate1(i,2)==mate2(i,2);
        tem2=ones(nclassifiers,1);
        tem2(ind1,1)=zeros(length(ind1),1);
        eval(['[ind2,tem1]=create(CS',type2,'(tem2,:),sum(tem2),l2,',  ..
              'condition2);'])
        eval(['CS',type2,'(tem2,:)=tem1;'])
        ind2=find(cumsum(tem2)==ind2);
        ind2=ind2(1);
      else;
        eval(['[ind2,CS',type2,']=create(CS',type2,',nclassifiers,l2,'  ..
              'condition2);'])
      end;
    end;
    %
    %  (d) find matching classifiers with winning bids ..
    %
    eval(['c1=[ind1,CS',type1,'(ind1,l2+3)];'])
    eval(['c2=[ind2,CS',type2,'(ind2,l2+3)];'])
    win1=find(c1(:,2)>=max(c1(:,2)));
    win2=find(c2(:,2)>=max(c2(:,2)));
    win1=c1(win1(ceil(rand*length(win1))),:); 
    win2=c2(win2(ceil(rand*length(win2))),:);
    %
    %  (e) and their strings,
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
    %  (f) find next period's storage and return for each type,
    %
    arg=4*l+4;
    tem1=ones(6*ntypes,1)*[string2,string1];
    eval(['ind3=find(~sum(abs(((R',type1,'(:,1:arg)>=0 & tem1>=0).*R',type1, ..
          '(:,1:arg)+(R',type1,'(:,1:arg)<0 | tem1<0).*tem1-tem1)'')));'])
    eval(['next=R',type1,'(ind3(1),arg+1:arg+l+1);'])
    popstorage(mate1(i,1)+nagents*(mate1(i,2)-1),:)=next(1:l);
    eval(['CS',type1,'(win1(1),l2+3)=CS',type1,'(win1(1),l2+3)*(1-bid1', ..
          '(mate1(i,2))-bid2(mate1(i,2))*sp1)+next(l+1)-tax(mate1(i,2));'])

    tem1=ones(6*ntypes,1)*[string1,string2];
    eval(['ind4=find(~sum(abs(((R',type2,'(:,1:arg)>=0 & tem1>=0).*R',type2, ..
          '(:,1:arg)+(R',type2,'(:,1:arg)<0 | tem1<0).*tem1-tem1)'')));'])
    eval(['next=R',type2,'(ind4(1),arg+1:arg+l+1);'])
    popstorage(mate2(i,1)+nagents*(mate2(i,2)-1),:)=next(1:l);
    eval(['CS',type2,'(win2(1),l2+3)=CS',type2,'(win2(1),l2+3)*(1-bid1', ..
          '(mate2(i,2))-bid2(mate2(i,2))*sp2)+next(l+1)-tax(mate2(i,2));'])
    %
    %  (g) update the number of times the rule was called and the number
    %      of exchanges,
    %
    eval(['CS',type1,'(win1(1),l2+4:l2+5)=CS',type1,'(win1(1),l2+4:l2+5)', ..
          '+[(R',type1,'(ind3(1),l2+1) & R',type1,'(ind3(1),4*l+3)),1];'])
    eval(['CS',type2,'(win2(1),l2+4:l2+5)=CS',type2,'(win2(1),l2+4:l2+5)', ..
          '+[(R',type2,'(ind4(1),l2+1) & R',type2,'(ind4(1),4*l+3)),1];'])
    %
    %  (h) and rescale strengths.
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
  %  For every Tga periods, run the genetic algorithm.
  %
  if rem(it,Tga)==0;
    for i=1:ntypes;
      fprintf('Genetic Algorithm for Classifier System %g \n',i)
      eval(['CS',int2str(i),'=ga(CS',int2str(i),',nselect,pcross', ..
       ',pmutation,crowdingfactor,crowdingsubpop,nclassifiers,l2,smultiple);'])
    end;  
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
    fprintf('  Histogram for Type %g Agents: \n',i)
    disp('  ----------------------------')
    disp(' ')
    for j=1:ntypes;
      no=sum(~sum(abs( (popstorage(nagents*(i-1)+1:nagents*i,:)-  ..
         ones(nagents,1)*bnames(j,:))' )));
      fprintf('  Number holding good %g: %g \n',j,no);
    end;
    disp(' ')
    if rem(it,dispclass)==0;
      disp(' ')
      fprintf('  Classifier System for Type %g Agents: \n',i)
      disp('  -------------------------------------')
      disp(' ')
      eval(['disp([[1:nclassifiers]'',CS',int2str(i),',[1:nclassifiers]''])'])
    end;
  end;
end;
