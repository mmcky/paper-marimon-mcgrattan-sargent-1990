%  A Simple Classifier System applied to Wicksell N-tangles
%  --------------------------------------------------------
%
%
%  Initializations:
%
%
%   (a) initialize parameters
%
winitial                          % contains list of parameters
total=sum(nagents);
if rem(total,2)>0; 
  error('The total population of the economy must be an even number');
end;
[row,l]=size(bnames);
l2=2*l;
nselectt=round(propselectt*nclasst*.5);
nselectc=round(propselectc*nclassc*.5);
smultiple=2;                      % smultiple is a parameter of scale.m
%
%   (b) initialize the classifier systems CSt<i>, and CSc<i>, and
%       matrices for each type i with the following elements:
%
%         freq1(i,j,k) = # of times type i starts with good j and ends with k
%         freq2(i,j)   = # of times type i holds good j
%         freq3(i,j,k) = # of times type i carrying good j meets someone with k
%         freq4(i,j,k) = # of times type i carrying good j meets someone with k 
%                        and trade occurs
%         freq5(i,j)   = # of times type i carrying good j consumes it
%
%       Note: these frequencies will be used to calculate:
%  
%         prob1(i,j,k) = probability that a type i holding good j will be
%                        holding good k in the next period 
%                      = freq1(i,j,k) / freq2(i,j)
%                        ( transition )
%         prob2(i,j,k) = probability that a type i holding good j will meet
%                        someone holding good k
%                      = freq3(i,j,k) / freq2(i,j)
%                        ( meeting )
%         prob3(i,j,k) = probability that trade occurs when a type i holding
%                        good j meets someone holding k
%                      = freq4(i,j,k) / freq3(i,j,k)
%                        ( trading )
%         prob4(i,j)   = probability that consumption occurs when a type i
%                        is holding good j
%                      = freq5(i,j) / freq2(i,j)
%                        ( consuming )
%
%
shist=[ ];
for i=1:ntypes;
  k=int2str(i);
  eval(['sfreq1',k,'=[ ];'])
  eval(['sfreq2',k,'=[ ];'])
  eval(['sfreq3',k,'=[ ];'])
  eval(['sfreq4',k,'=[ ];'])
  eval(['sfreq5',k,'=[ ];'])
  eval(['sfreq1s',k,'=[ ];'])
  eval(['sfreq2s',k,'=[ ];'])
  eval(['sfreq3s',k,'=[ ];'])
  eval(['sfreq4s',k,'=[ ];'])
  eval(['sfreq5s',k,'=[ ];'])
  for j=1:ntypes;
    l=int2str(j);
    eval(['sclassc',k,l,'=[ ];'])
    for ii=1:ntypes;
      m=int2str(ii);
      eval(['sclasst',k,l,m,'=[ ];'])
    end;
  end;
end;
probt(3,:)=1-sum(probt);
probc(3,:)=1-sum(probc);
cst=cumsum(probt);
csc=cumsum(probc);
ntype2=ntypes*ntypes;
for i=1:ntypes;
  k=int2str(i);
  tem1=[ ]; tem2=[ ];
  for j=1:nclasst;
    tem1=[tem1;sum(ones(3,1)*rand(1,l2)-cst>0)-1];
  end;
  for j=1:nclassc;
    tem2=[tem2;sum(ones(3,1)*rand(1,l)-csc>0)-1];
  end;
  eval(['CSt',k,'=[tem1,round(rand(nclasst,1)),strengtht(:,i),', ..
        'zeros(nclasst,3)];'])
  eval(['CSc',k,'=[tem2,round(rand(nclassc,1)),strengthc(:,i),', ..
        'zeros(nclassc,2)];'])
  eval(['freq1',k,'=zeros(ntypes);'])
  eval(['freq2',k,'=zeros(ntypes,1);'])
  eval(['freq3',k,'=zeros(ntypes);'])
  eval(['freq4',k,'=zeros(ntypes);'])
  eval(['freq5',k,'=zeros(ntypes,1);'])
  eval(['freq1s',k,'=zeros(ntypes);'])
  eval(['freq2s',k,'=zeros(ntypes,1);'])
  eval(['freq3s',k,'=zeros(ntypes);'])
  eval(['freq4s',k,'=zeros(ntypes);'])
  eval(['freq5s',k,'=zeros(ntypes,1);'])
  eval(['freq1st',k,'=zeros(nback,ntype2);'])
  eval(['freq2st',k,'=zeros(nback,ntypes);'])
  eval(['freq3st',k,'=zeros(nback,ntype2);'])
  eval(['freq4st',k,'=zeros(nback,ntype2);'])
  eval(['freq5st',k,'=zeros(nback,ntypes);'])
end;
%  A test:
eval(rules);
CSt1=[rule1,zeros(72,4)];
CSt2=CSt1;
CSt3=CSt1;
CSc1=[rule2,zeros(12,3)];
CSc2=CSc1;
CSc3=CSc1;
%
%  (c) initialize the storages for the population
%
%disp('*** POPSTORAGE AND CLASSIFIERS TAMPERED WITH')
%tem1=[0 0;1 0;0 1;1 1];
popstorage=bnames(1+floor(rand(total,1)*ntypes),:);
%
%  (d) initialize the indices for winning classifiers of the iteration before
%      who will be rewarded bid payments
%
last1=zeros(max(nagents),ntypes);                
last2=zeros(max(nagents),ntypes);                
%
%  (e) initialize vectors initgat and initgac for controlling printing
%      of classifiers during inital genetic algorithm calls.
%
initgat=ones(ntypes,1);
initgac=initgat;
%
%  (f) and print out parameters and original classifier systems.
%
disp(' ')
disp('Parameter Specifications')
disp('------------------------')
disp(' ')
fprintf(' Number of agent types = %g\n',ntypes)
fprintf(' Number of strings in trade classifier systems = %g\n',nclasst)
fprintf(' Number of strings in consume classifier systems = %g\n',nclassc)
string=' Number of periods between displays of trade classifiers = %g\n';
fprintf(string,dclasst)
string=' Number of periods between displays of consume classifiers = %g\n';
fprintf(string,dclassc)
string=' Number of periods between displays of histograms = %g\n';
fprintf(string,dhist)
string=' Number of periods between displays of probability matrices = %g\n';
fprintf(string,dprob)
%string=[' Number of periods between genetic algorithm calls for trade ', ..
%         'classifiers = %g\n'];
%fprintf(string,Tgat)
%string=[' Number of periods between genetic algorithm calls for consume ', ..
%         'classifiers = %g\n'];
%fprintf(string,Tgac)
disp(' Iterations with genetic algorithm call given in RUNIT')
string=[' Proportion of strings in trade classifier chosen for ', ..
         'reproduction = %g\n'];
fprintf(string,propselectt)
string=[' Proportion of strings in consume classifier chosen for ', ..
         'reproduction = %g\n'];
fprintf(string,propselectc)
fprintf(' Probability of crossover for trade classifier = %g\n',pcrosst)
fprintf(' Probability of crossover for consume classifier = %g\n',pcrossc)
fprintf(' Probability of mutation for trade classifier = %g\n',pmutationt)
fprintf(' Probability of mutation for consume classifier = %g\n',pmutationc)
fprintf(' Crowding subpopulation for trade classifier = %g\n',crowdsubpopt)
fprintf(' Crowding subpopulation for consume classifier = %g\n',crowdsubpopc)
fprintf(' Crowding factor for trade classifier = %g\n',crowdfactort)
fprintf(' Crowding factor for consume classifier = %g\n',crowdfactorc)
disp(' Number of agents for each type:')
disp(nagents)
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
disp(' Bid1 for trade classifier:')
disp(tbid1)
disp(' Bid2 for trade classifier:')
disp(tbid2)
disp(' Bid1 for consume classifier:')
disp(cbid1)
disp(' Bid2 for consume classifier:')
disp(cbid2)
disp(' Taxes:')
disp(tax)
disp(' Probabilities of -1,0 in generating trade classifier strings:')
disp(probt)
disp(' Probabilities of -1,0 in generating consume classifier strings:')
disp(probc)
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
  eval(['disp([[1:nclasst]'',CSt',int2str(i),',[1:nclasst]''])'])
  eval(['disp([[1:nclassc]'',CSc',int2str(i),',[1:nclassc]''])'])
  disp(' ')
end;
%
%
list=[ ];
for i=1:ntypes;
  list=[list; [1:nagents(i)]',ones(nagents(i),1)*i];
end;
%
%  For maxit iterations, 
%
for it=1:maxit
  for i=1:ntypes;
    k=int2str(i);
    eval(['Tax(:,i)=tax(i)*abs(CSt',k,'(:,l2+2));'])
    eval(['freq1st',k,'=[freq1st',k,'(2:nback,:);zeros(1,ntype2)];'])
    eval(['freq2st',k,'=[freq2st',k,'(2:nback,:);zeros(1,ntypes)];'])
    eval(['freq3st',k,'=[freq3st',k,'(2:nback,:);zeros(1,ntype2)];'])
    eval(['freq4st',k,'=[freq4st',k,'(2:nback,:);zeros(1,ntype2)];'])
    eval(['freq5st',k,'=[freq5st',k,'(2:nback,:);zeros(1,ntypes)];'])
  end;
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
    condition1=[popstorage(mate1(i,1)+sum(nagents(1:mate1(i,2)-1)),:), ..
                popstorage(mate2(i,1)+sum(nagents(1:mate2(i,2)-1)),:)];
    condition2=condition1([l+1:l2,1:l]);
    %
    % (b) get strings type1 and type2 giving agent types,
    %
    type1=int2str(mate1(i,2));
    type2=int2str(mate2(i,2));
    %
    % (c) update freq3<i>(j,k) and freq2<i>(j)
    %
    %
    ind1=find(~sum(abs( (bnames-ones(ntypes,1)*condition1(1:l))' )));
    ind2=find(~sum(abs( (bnames-ones(ntypes,1)*condition2(1:l))' )));
    eval(['freq3',type1,'(ind1,ind2)=freq3',type1,'(ind1,ind2)+1;'])
    eval(['freq3',type2,'(ind2,ind1)=freq3',type2,'(ind2,ind1)+1;'])
    fstr=['freq3st',type1,'(nback,ntypes*(ind2-1)+ind1)'];
    eval([fstr,'=',fstr,'+1;'])
    fstr=['freq3st',type2,'(nback,ntypes*(ind1-1)+ind2)'];
    eval([fstr,'=',fstr,'+1;'])
    eval(['freq2',type1,'(ind1,1)   =freq2',type1,'(ind1,1)+1;'])
    eval(['freq2',type2,'(ind2,1)   =freq2',type2,'(ind2,1)+1;'])
    fstr=['freq2st',type1,'(nback,ind1)'];
    eval([fstr,'=',fstr,'+1;'])
    fstr=['freq2st',type2,'(nback,ind2)'];
    eval([fstr,'=',fstr,'+1;'])
    %
    % (d) find indices of classifiers in CSt matching conditions and
    %     if there are no matches, replace a string with the condition,
    %
    cstr=['CSt',type1,'(:,1:l2)'];
    eval(['ind3=find(~sum(abs(( (',cstr,'>=0).*',cstr,'+(',cstr,'<0).*(ones', ..
      '(nclasst,1)*condition1)-ones(nclasst,1)*condition1)'')))'';'])
    if isempty(ind3); 
      eval(['[ind3,CSt',type1,']=create(CSt',type1,',nclasst,l2,'  ..
            'condition1);'])
      eval(['CSt',type1,'(ind3,l2+5)=it;'])
    end;

    cstr=['CSt',type2,'(:,1:l2)'];
    eval(['ind4=find(~sum(abs(( (',cstr,'>=0).*',cstr,'+(',cstr,'<0).*(ones', ..
      '(nclasst,1)*condition2)-ones(nclasst,1)*condition2)'')))'';'])
    if isempty(ind4);
      if mate1(i,2)==mate2(i,2);
        tem2=ones(nclasst,1);
        tem2(ind3,1)=zeros(length(ind3),1);
        eval(['[ind4,tem1]=create(CSt',type2,'(tem2,:),sum(tem2),l2,',  ..
              'condition2);'])
        eval(['CSt',type2,'(tem2,:)=tem1;'])
        ind4=find(cumsum(tem2)==ind4);
        ind4=ind4(1);
      else;
        eval(['[ind4,CSt',type2,']=create(CSt',type2,',nclasst,l2,'  ..
              'condition2);'])
      end;
      eval(['CSt',type2,'(ind4,l2+5)=it;'])
    end;
    %
    % (e) find matching classifiers with winning bids ..
    %
    ind11=[1:nclasst]';
    ind11(ind3)=zeros(ind3);
    ind11=find(ind11);
    ind12=[1:nclasst]';
    ind12(ind4)=zeros(ind4);
    ind12=find(ind12);
    
    eval(['c1=[ind3,CSt',type1,'(ind3,l2+2)];'])
    eval(['c2=[ind4,CSt',type2,'(ind4,l2+2)];'])
    win1=find(c1(:,2)>=max(c1(:,2)));
    win2=find(c2(:,2)>=max(c2(:,2)));
    win1=c1(win1(1+floor(rand*length(win1))),:);
    win2=c2(win2(1+floor(rand*length(win2))),:);
    
    eval(['ind3=ind3(find(CSt',type1,'(ind3,l2+1)==CSt',type1,  ..
          '(win1(1),l2+1)));'])
    eval(['ind4=ind4(find(CSt',type2,'(ind4,l2+1)==CSt',type2,  ..
          '(win2(1),l2+1)));'])
    ind3=ind3(find(ind3~=win1(1)));
    ind4=ind4(find(ind4~=win2(1)));

    %
    % (f) and their strings,
    %
    eval(['string1=CSt',type1,'(win1(1),1:l2+1);'])
    eval(['string2=CSt',type2,'(win2(1),1:l2+1);'])
    sp1=sum(string1(1:l)<0)*isempty(find(~sum(abs(bnames-ones(ntypes,1)* ..
        string1(1:l))')))+sum(string1(l+1:l2)<0)*isempty(find(~sum(abs   ..
        (bnames-ones(ntypes,1)*string1(l+1:l2))')));
    sp1=1/(1+sp1);
    sp2=sum(string2(1:l)<0)*isempty(find(~sum(abs(bnames-ones(ntypes,1)* ..
        string2(1:l))')))+sum(string2(l+1:l2)<0)*isempty(find(~sum(abs   ..
        (bnames-ones(ntypes,1)*string2(l+1:l2))')));
    sp2=1/(1+sp2);

    % 
    % (h) determine if agents traded:
    % 
    eval(['trade1=(CSt',type1,'(win1(1),l2+1) );'])
    eval(['trade2=(CSt',type2,'(win2(1),l2+1) );'])
    trade=(trade1 & trade2);
    %
    % (i) get conditions: own storage     ... for CSc matrices
    %
    if trade;
      condition3=condition2(1:l);
      condition4=condition1(1:l);
      ind7=ind2;
      ind8=ind1;
    else;
      condition3=condition1(1:l);
      condition4=condition2(1:l);
      ind7=ind1;
      ind8=ind2;
    end;
    %
    % (g) reward bids to sources of current winners 
    %
    cbid=tbid1(mate1(i,2))+tbid2(mate1(i,2))*sp1;
    source=last1(mate1(i,1),mate1(i,2));
    if source>0;
      cstr=['CSc',type1,'(:,1:l)']; 
      tem1=bnames(last2(mate1(i,1),mate1(i,2)),:);
      eval(['ind13=find(~sum(abs(( (',cstr,'>=0).*',cstr,'+(',cstr,'<0).*', ..
            '(ones(nclassc,1)*tem1)-ones(nclassc,1)*tem1 )'')))'';'])
      ind15=[1:nclassc]';
      ind15(ind13)=zeros(ind13);
      ind15=find(ind15);
      eval(['ind13=ind13(find(CSc',type1,'(ind13,l+1)==CSc',type1,  ..
            '(source,l+1)));'])
      ind13=ind13(find(ind13~=source));
      if ( (trade1==trade2) | trade1==0 );
        eval(['tem1=cbid*CSt',type1,'(win1(1),l2+2);'])
        eval(['CSc',type1,'(source,l+2)=CSc',type1,'(source,l+2)+tem1;'])
        if ~isempty(ind13);
          eval(['CSc',type1,'(ind13,l+2)=CSc',type1,  ..
                '(ind13,l+2)+frac1*tem1;'])
        end;
        if ~isempty(ind15);
          eval(['CSc',type1,'(ind15,l+2)=CSc',type1,  ..
                '(ind15,l+2)+frac2*tem1;'])
        end;
      end;
    end;
    if ( (trade1==trade2) | trade1==0 );
      eval(['CSt',type1,'(win1(1),l2+2)=CSt',type1,'(win1(1),l2+2)*(1-cbid);'])
      if ~isempty(ind3);
        eval(['CSt',type1,'(ind3,l2+2)=CSt',type1,  ..
              '(ind3,l2+2)*(1-frac1*cbid);'])
      end;
      if ~isempty(ind11);
        eval(['CSt',type1,'(ind11,l2+2)=CSt',type1,  ..
              '(ind11,l2+2)*(1-frac2*cbid);'])
      end;
    end;

    cbid=tbid1(mate2(i,2))+tbid2(mate2(i,2))*sp2;
    source=last1(mate2(i,1),mate2(i,2));
    if source>0;
      cstr=['CSc',type2,'(:,1:l)'];
      tem1=bnames(last2(mate2(i,1),mate2(i,2)),:);
      eval(['ind14=find(~sum(abs(( (',cstr,'>=0).*',cstr,'+(',cstr,'<0).*', ..
            '(ones(nclassc,1)*tem1)-ones(nclassc,1)*tem1 )'')))'';'])
      ind16=[1:nclassc]';
      ind16(ind14)=zeros(ind14);
      ind16=find(ind16);
      eval(['ind14=ind14(find(CSc',type2,'(ind14,l+1)==CSc',type2,  ..
            '(source,l+1)));'])
      ind14=ind14(find(ind14~=source));
      if ( (trade1==trade2) | trade1==0 );
        eval(['tem1=cbid*CSt',type2,'(win2(1),l2+2);'])
        eval(['CSc',type2,'(source,l+2)=CSc',type2,'(source,l+2)+tem1;'])
        if ~isempty(ind14);
          eval(['CSc',type2,'(ind14,l+2)=CSc',type2,  ..
                '(ind14,l+2)+frac1*tem1;'])
        end;
        if ~isempty(ind16);
          eval(['CSc',type2,'(ind16,l+2)=CSc',type2,  ..
                '(ind16,l+2)+frac2*tem1;'])
        end;
      end;
    end;
    if ( (trade1==trade2) | trade1==0 );
      eval(['CSt',type2,'(win2(1),l2+2)=CSt',type2,'(win2(1),l2+2)*(1-cbid);'])
      if ~isempty(ind4);
        eval(['CSt',type2,'(ind4,l2+2)=CSt',type2,  ..
              '(ind4,l2+2)*(1-frac1*cbid);'])
      end;
      if ~isempty(ind12);
        eval(['CSt',type2,'(ind12,l2+2)=CSt',type2,  ..
            '(ind12,l2+2)*(1-frac2*cbid);'])
      end;
    end;
    %
    % (j) find indices of classifiers in CSc matching conditions and
    %     if there are no matches, replace a string with the condition,
    %
    cstr=['CSc',type1,'(:,1:l)'];
    eval(['ind5=find(~sum(abs(( (',cstr,'>=0).*',cstr,'+(',cstr,'<0).*(ones', ..
      '(nclassc,1)*condition3)-ones(nclassc,1)*condition3)'')))'';'])
    if isempty(ind5); 
      eval(['[ind5,CSc',type1,']=create(CSc',type1,',nclassc,l,'  ..
            'condition3);'])
      eval(['CSc',type1,'(ind5,l+4)=it;'])
      tem1=find(~(last(:,mate1(i,2))-ind5));
      if ~isempty(tem1); last(tem1,mate1(i,2))=zeros(length(tem1),1); end;
    end;

    cstr=['CSc',type2,'(:,1:l)'];
    eval(['ind6=find(~sum(abs(( (',cstr,'>=0).*',cstr,'+(',cstr,'<0).*(ones', ..
      '(nclassc,1)*condition4)-ones(nclassc,1)*condition4)'')))'';'])
    if isempty(ind6);
      if mate1(i,2)==mate2(i,2);
        tem2=ones(nclassc,1);
        tem2(ind5,1)=zeros(length(ind5),1);
        eval(['[ind6,tem1]=create(CSc',type2,'(tem2,:),sum(tem2),l,',  ..
              'condition4);'])
        eval(['CSc',type2,'(tem2,:)=tem1;'])
        ind6=find(cumsum(tem2)==ind6);
        ind6=ind6(1);
      else;
        eval(['[ind6,CSc',type2,']=create(CSc',type2,',nclassc,l,'  ..
              'condition4);'])
      end;
      eval(['CSc',type2,'(ind6,l+4)=it;'])
      tem1=find(~(last(:,mate2(i,2))-ind6));
      if ~isempty(tem1); last(tem1,mate2(i,2))=zeros(length(tem1),1); end;
    end;
    %
    % (k) find matching classifiers with winning bids ..
    %

    ind17=[1:nclassc]';
    ind17(ind5)=zeros(ind5);
    ind17=find(ind17);
    ind18=[1:nclassc]';
    ind18(ind6)=zeros(ind6);
    ind18=find(ind18);

    eval(['c1=[ind5,CSc',type1,'(ind5,l+2)];'])
    eval(['c2=[ind6,CSc',type2,'(ind6,l+2)];'])
    win3=find(c1(:,2)>=max(c1(:,2)));
    win4=find(c2(:,2)>=max(c2(:,2)));
    win3=c1(win3(1+floor(rand*length(win3))),:);
    win4=c2(win4(1+floor(rand*length(win4))),:);

    eval(['ind5=ind5(find(CSc',type1,'(ind5,l+1)==CSc',type1,  ..
          '(win3(1),l+1)));'])
    eval(['ind6=ind6(find(CSc',type2,'(ind6,l+1)==CSc',type2,  ..
          '(win4(1),l+1)));'])
    ind5=ind5(find(ind5~=win3(1)));
    ind6=ind6(find(ind6~=win4(1)));

    %
    % (l) and their strings,
    %
    eval(['string1=CSc',type1,'(win3(1),1:l+1);'])
    eval(['string2=CSc',type2,'(win4(1),1:l+1);'])
    sp1=sum(string1(1:l)<0)*isempty(find(~sum(abs(bnames-ones(ntypes,1)* ..
        string1(1:l))')));
    sp1=1/(1+sp1);
    sp2=sum(string2(1:l)<0)*isempty(find(~sum(abs(bnames-ones(ntypes,1)* ..
        string2(1:l))')));
    sp2=1/(1+sp2);

    %
    % (m) reward bids to sources of current winners 
    %
    eval(['cons1=CSc',type1,'(win3(1),l+1);'])
    eval(['cons2=CSc',type2,'(win4(1),l+1);'])

    cbid=cbid1(mate1(i,2))+cbid2(mate1(i,2))*sp1;
    if ( (trade1==trade2) | trade1==0 );
      eval(['tem1=cbid*CSc',type1,'(win3(1),l+2);'])
      eval(['CSt',type1,'(win1(1),l2+2)=CSt',type1,'(win1(1),l2+2)+tem1;'])
      if ~isempty(ind3);
        eval(['CSt',type1,'(ind3,l2+2)=CSt',type1,'(ind3,l2+2)+frac1*tem1;'])
      end;
      if ~isempty(ind11);
        eval(['CSt',type1,'(ind11,l2+2)=CSt',type1,'(ind11,l2+2)+frac2*tem1;'])
      end;
    end;
    eval(['tem1=cons1*(( ind7==mate1(i,2) )*utility(mate1(i,2))-',  ..
          'prodcosts(mate1(i,2)) )-storecosts(mate1(i,2),ind7)-', ..
          'cbid*CSc',type1,'(win3(1),l+2);'])
    eval(['CSc',type1,'(win3(1),l+2)=CSc',type1,'(win3(1),l+2)+tem1;'])
    if ~isempty(ind5);
      eval(['CSc',type1,'(ind5,l+2)=CSc',type1,'(ind5,l+2)+frac1*tem1;'])
    end;
    if ~isempty(ind17);
      eval(['CSc',type1,'(ind17,l+2)=CSc',type1,'(ind17,l+2)+frac2*tem1;'])
    end;
    last1(mate1(i,1),mate1(i,2))=win3(1);
    last2(mate1(i,1),mate1(i,2))=ind7;

    cbid=cbid1(mate2(i,2))+cbid2(mate2(i,2))*sp2;
    if ( (trade1==trade2) | trade1==0 );
      eval(['tem1=cbid*CSc',type1,'(win4(1),l+2);'])
      eval(['CSt',type2,'(win2(1),l2+2)=CSt',type2,'(win2(1),l2+2)+tem1;'])
      if ~isempty(ind4);
        eval(['CSt',type2,'(ind4,l2+2)=CSt',type2,'(ind4,l2+2)+frac1*tem1;'])
      end;
      if ~isempty(ind12);
        eval(['CSt',type2,'(ind12,l2+2)=CSt',type2,'(ind12,l2+2)+frac2*tem1;'])
      end;
    end;
    eval(['tem1=cons2*(( ind8==mate2(i,2) )*utility(mate2(i,2))-',  ..
          'prodcosts(mate2(i,2)) )-storecosts(mate2(i,2),ind8)-', ..
          'cbid*CSc',type2,'(win4(1),l+2);'])
    eval(['CSc',type2,'(win4(1),l+2)=CSc',type2,'(win4(1),l+2)+tem1;'])
    if ~isempty(ind6);
      eval(['CSc',type2,'(ind6,l+2)=CSc',type2,'(ind6,l+2)+frac1*tem1;'])
    end;
    if ~isempty(ind18);
      eval(['CSc',type2,'(ind18,l+2)=CSc',type2,'(ind18,l+2)+frac2*tem1;'])
    end;
    last1(mate2(i,1),mate2(i,2))=win4(1);
    last2(mate2(i,1),mate2(i,2))=ind8;

    if cons1;
      ind9=produces(mate1(i,2));
      popstorage(mate1(i,1)+sum(nagents(1:mate1(i,2)-1)),:)=bnames(ind9,:);
    else;
      ind9=ind7;
      popstorage(mate1(i,1)+sum(nagents(1:mate1(i,2)-1)),:)=condition3;
    end;

    if cons2;
      ind10=produces(mate2(i,2));
      popstorage(mate2(i,1)+sum(nagents(1:mate2(i,2)-1)),:)=bnames(ind10,:);
    else;
      ind10=ind8;
      popstorage(mate2(i,1)+sum(nagents(1:mate2(i,2)-1)),:)=condition4;
    end;


    %
    % (h) update the number of times the rule was called and the number
    %      of exchanges,
    %
    eval(['CSt',type1,'(win1(1),l2+3:l2+4)=CSt',type1,'(win1(1),l2+3:l2+4)', ..
          '+[trade,1];'])
    eval(['CSt',type2,'(win2(1),l2+3:l2+4)=CSt',type2,'(win2(1),l2+3:l2+4)', ..
          '+[trade,1];'])
    eval(['CSc',type1,'(win3(1),l+3)=CSc',type1,'(win3(1),l+3)+1;'])
    eval(['CSc',type2,'(win4(1),l+3)=CSc',type2,'(win4(1),l+3)+1;'])

    %
    % (i) update the freq1<i>, freq4<i>, and freq5<i> matrices
    %
    eval(['freq1',type1,'(ind1,ind9)=freq1',type1,'(ind1,ind9)+1;'])
    eval(['freq1',type2,'(ind2,ind10)=freq1',type2,'(ind2,ind10)+1;'])
    fstr=['freq1st',type1,'(nback,ntypes*(ind9-1)+ind1)'];
    eval([fstr,'=',fstr,'+1;'])
    fstr=['freq1st',type2,'(nback,ntypes*(ind10-1)+ind2)'];
    eval([fstr,'=',fstr,'+1;'])
    eval(['freq4',type1,'(ind1,ind2)=freq4',type1,'(ind1,ind2)+trade;'])
    eval(['freq4',type2,'(ind2,ind1)=freq4',type2,'(ind2,ind1)+trade;'])
    fstr=['freq4st',type1,'(nback,ntypes*(ind2-1)+ind1)'];
    eval([fstr,'=',fstr,'+trade;'])
    fstr=['freq4st',type2,'(nback,ntypes*(ind1-1)+ind2)'];
    eval([fstr,'=',fstr,'+trade;'])
    eval(['freq5',type1,'(ind1,1)=freq5',type1,'(ind1,1)+cons1;'])
    eval(['freq5',type2,'(ind2,1)=freq5',type2,'(ind2,1)+cons2;'])
    fstr=['freq5st',type1,'(nback,ind1)'];
    eval([fstr,'=',fstr,'+cons1;'])
    fstr=['freq5st',type2,'(nback,ind2)'];
    eval([fstr,'=',fstr,'+cons2;'])
  end;
  %
  %  Subtract off taxes computed at the beginning of the iteration and
  %  Derive frequency matrices for T-nback to T.
  %
  for i=1:ntypes;
    k=int2str(i);
    eval(['CSt',k,'(:,l2+2)=CSt',k,'(:,l2+2)-Tax(:,i);'])
    eval(['freq1s',k,'(:)=sum(freq1st',k,');'])
    eval(['freq2s',k,'=sum(freq2st',k,')'';'])
    eval(['freq3s',k,'(:)=sum(freq3st',k,');'])
    eval(['freq4s',k,'(:)=sum(freq4st',k,');'])
    eval(['freq5s',k,'=sum(freq5st',k,')'';'])
  end;
  %
  %  Store certain information
  %
  if ~rem(it,savef);
    for i=1:ntypes;
      k=int2str(i);
      eval(['sfreq1',k,'=[sfreq1',k,';freq1',k,'(:)''];'])
      eval(['sfreq2',k,'=[sfreq2',k,';freq2',k,'(:)''];'])
      eval(['sfreq3',k,'=[sfreq3',k,';freq3',k,'(:)''];'])
      eval(['sfreq4',k,'=[sfreq4',k,';freq4',k,'(:)''];'])
      eval(['sfreq5',k,'=[sfreq5',k,';freq5',k,'(:)''];'])
      eval(['sfreq1s',k,'=[sfreq1s',k,';freq1s',k,'(:)''];'])
      eval(['sfreq2s',k,'=[sfreq2s',k,';freq2s',k,'(:)''];'])
      eval(['sfreq3s',k,'=[sfreq3s',k,';freq3s',k,'(:)''];'])
      eval(['sfreq4s',k,'=[sfreq4s',k,';freq4s',k,'(:)''];'])
      eval(['sfreq5s',k,'=[sfreq5s',k,';freq5s',k,'(:)''];'])
    end;
  end;
  if ~rem(it,savec);
    for i=1:ntypes;
      k=int2str(i);
      for j=1:ntypes;
        l=int2str(j);
        for ii=1:ntypes;
          m=int2str(ii);
          cstr=['CSt',k,'(:,1:l2)'];
          tem1=[bnames(j,:),bnames(ii,:)];
          eval(['tem2=find(~sum(abs(( (',cstr,'>=0).*',cstr,'+(',cstr, ..
                '<0).*(ones(nclasst,1)*tem1)-ones(nclasst,1)*tem1)'')))'';'])
          if ~isempty(tem2); 
            eval(['[tem1,tem3]=sort(CSt',k,'(tem2,l2+2));'])
            tem4=length(tem3);
            eval(['sclasst',k,l,m,'=[sclasst',k,l,m,';CSt',k,  ..
                  '(tem3(tem4),1:10)];')
            if tem4>1;
              eval(['sclasst',k,l,m,'=[sclasst',k,l,m,';CSt',k,  ..
                    '(tem3(tem4-1),1:10)];')
            else;
              eval(['sclasst',k,l,m,'=-2*ones(1,10);'])
            end;
          else;
            eval(['sclasst',k,l,m,'=-2*ones(2,10);'])
          end;
        end;
        cstr=['CSc',k,'(:,1:l)'];
        tem1=bnames(j,:);
        eval(['tem2=find(~sum(abs(( (',cstr,'>=0).*',cstr,'+(',cstr, ..
              '<0).*(ones(nclassc,1)*tem1)-ones(nclassc,1)*tem1)'')))'';'])
        if ~isempty(tem2); 
          eval(['[tem1,tem3]=sort(CSc',k,'(tem2,l+2));'])
          tem4=length(tem3);
          eval(['sclassc',k,l,'=[sclassc',k,l,';CSc',k,  ..
                '(tem3(tem4),1:6)];')
          if tem4>1;
            eval(['sclassc',k,l,'=[sclassc',k,l,';CSc',k,  ..
                  '(tem3(tem4-1),1:6)];')
          else;
            eval(['sclassc',k,l,'=-2*ones(1,6);'])
          end;
        else;
          eval(['sclassc',k,l,'=-2*ones(2,6);'])
        end;
      end;
    end;
  end;
  %
  %  Print out the results for iteration "it".
  %
  if (~rem(it,dhist)   | ~rem(it,dprob) | ~rem(it,dclasst) |  ..
       runit(it)==1 )
%      ~rem(it,dclassc) | ~rem(it,Tgat)  | ~rem(it,Tgac) )
    disp(' ')
    fprintf('Results for Iteration %g: \n',it)
    disp('-------------------------')
  end;
  if ~rem(it,saveh) | ~rem(it,dhist);
    tem2=[ ];
    for i=1:ntypes;
      k=int2str(i);
      tem1=[ ];
      for j=1:ntypes;
        no=sum(~sum(abs( (popstorage(sum(nagents(1:i-1))+1:    ...
           sum(nagents(1:i)),:)-ones(nagents(i),1)*bnames(j,:))' )));
        tem1=[tem1;no];
      end;
      tem2=[tem2,tem1'];
      if ~rem(it,dhist);
        disp(' ')
        fprintf('  Histogram for Type %g Agents and Transition Matrix:\n',i)
        disp('  -------------------------------------------------------')
        eval([' disp([tem1,freq1',k,'./(freq2',k,'*ones(1,ntypes)),freq1s', ..
                k,'./(freq2s',k,'*ones(1,ntypes))]) '])
      end;
    end;
    if ~rem(it,saveh); shist=[shist;tem2]; end;
  end;
  if ~rem(it,dprob);
    for i=1:ntypes;
      k=int2str(i);
      disp(' ')
      fprintf('  Pr(holding i, meeting one with j) for Type %g Agents: \n',i)
      disp('  --------------------------------------------------------')
      eval(['disp([freq3',k,'./(freq2',k,'*ones(1,ntypes)),freq3s',  ..
             k,'./(freq2s',k,'*ones(1,ntypes))] )'])
      disp(' ')
      fprintf(['  Pr(trade | holding i, meeting one with j) for Type %g', ..
                ' Agents: \n'],i)
      disp('  ----------------------------------------------------------------')
      eval(['disp([freq4',k,'./freq3',k,',freq4s',k,'./freq3s',k,'])'])
      disp(' ')
      fprintf('  Pr(consumption | holding i) for Type %g Agents: \n',i)
      disp('  --------------------------------------------------')
      eval(['disp([freq5',k,'./freq2',k,',freq5s',k,'./freq2s',k,'])'])
      disp(' ')
    end;
  end;
  %
  %  For every Tgat periods, run the genetic algorithm for CSt.
  %
%  if ~rem(it,Tgat);
  if runit(it)==1;
    send=1+floor(rand*ntypes);
    if psecond>rand; 
      tem1=[1:ntypes]';
      tem1=tem1(tem1~=send);
      send=[send;tem1(1+floor(rand*(ntypes-1)))]; 
      if pthird>rand;
        tem1=tem1(tem1~=send(2));
        send=[send;tem1(1+floor(rand*(ntypes-2)))];
      end;
    end;
    nsend=length(send);

    if ~rem(it,dclasst);
      for i=1:ntypes;
        k=int2str(i);
        disp(' ')
        s='  Classifier System for Type %g Agents before Genetic Algorithm: \n';
        fprintf(s,i)
        disp('  --------------------------------------------------------------')
        disp(' ')
        eval(['disp([[1:nclasst]'',CSt',int2str(i),',[1:nclasst]''])'])
      end;
    end;
    if sum(initgat);
      for i=1:nsend;
        if initgat(send(i));
          k=int2str(send(i));
          disp(' ')
          s=['  Classifier System for Type %g Agents before Genetic', ..
              ' Algorithm: \n'];
          fprintf(s,send(i))
          disp(['  -------------------------------------------------',  ..
                   '-------------'])
          disp(' ')
          eval(['disp([[1:nclasst]'',CSt',k,',[1:nclasst]''])'])
        end;
      end;
    end;
    for i=1:nsend;
      k=int2str(send(i));
      fprintf('Genetic Algorithm for Classifier System %g \n',send(i))
      if exp1==1;
        eval(['[CSt',k,',last]=ga2(CSt',k,',nselectt,pcrosst', ..
         ',pmutationt,crowdfactort,crowdsubpopt,nclasst,l2', ..
         ',smultiple,last,',k,',it,l2+5,propmostusedt);'])
      else;
        eval(['[CSt',k,',last]=ga(CSt',k,',nselectt,pcrosst', ..
         ',pmutationt,crowdfactort,crowdsubpopt,nclasst,l2', ..
         ',smultiple,last,',k,',it,l2+5);'])
      end;
    end;
    if ~rem(it,dclasst);  
      for i=1:ntypes;
        disp(' ')
        s='  Classifier Systems for Type %g Agents after Genetic Algorithm: \n';
        fprintf(s,i)
        disp('  --------------------------------------------------------------')
        disp(' ')
        eval(['disp([[1:nclasst]'',CSt',int2str(i),',[1:nclasst]''])'])
      end;
    end;
    if sum(initgat);
      for i=1:nsend;
        if initgat(send(i));
          k=int2str(send(i));
          disp(' ')
          s=['  Classifier System for Type %g Agents after Genetic', ..
              ' Algorithm: \n'];
          fprintf(s,send(i))
          disp(['  -------------------------------------------------',  ..
                   '-------------'])
          disp(' ')
          eval(['disp([[1:nclasst]'',CSt',k,',[1:nclasst]''])'])
          initgat(send(i))=0;
        end;
      end;
    end;
  end;
  %
  %  For every Tgac periods, run the genetic algorithm for CSc.
  %
%  if ~rem(it,Tgac);
  if runit(it)==1;
    send=1+floor(rand*ntypes);
    if psecond>rand; 
      tem1=[1:ntypes]';
      tem1=tem1(tem1~=send);
      send=[send;tem1(1+floor(rand*(ntypes-1)))]; 
      if pthird>rand;
        tem1=tem1(tem1~=send(2));
        send=[send;tem1(1+floor(rand*(ntypes-2)))];
      end;
    end;
    nsend=length(send);

    if ~rem(it,dclassc);
      for i=1:ntypes;
        k=int2str(i);
        disp(' ')
        s='  Classifier System for Type %g Agents before Genetic Algorithm: \n';
        fprintf(s,i)
        disp('  --------------------------------------------------------------')
        disp(' ')
        eval(['disp([[1:nclassc]'',CSc',int2str(i),',[1:nclassc]''])'])
      end;
    end;
    if sum(initgac);
      for i=1:nsend;
        if initgac(send(i));
          k=int2str(send(i));
          disp(' ')
          s=['  Classifier System for Type %g Agents before Genetic', ..
              ' Algorithm: \n'];
          fprintf(s,send(i))
          disp(['  -------------------------------------------------',  ..
                   '-------------'])
          disp(' ')
          eval(['disp([[1:nclassc]'',CSc',k,',[1:nclassc]''])'])
        end;
      end;
    end;
    for i=1:nsend;
      k=int2str(send(i));
      fprintf('Genetic Algorithm for Classifier System %g \n',send(i))
      if exp1==1;
        eval(['[CSc',k,',last]=ga2(CSc',k,',nselectc,pcrossc', ..
         ',pmutationc,crowdfactorc,crowdsubpopc,nclassc,l', ..
         ',smultiple,last,',k,',it,l+4,propmostusedc);'])
      else;
        eval(['[CSc',k,',last]=ga(CSc',k,',nselectc,pcrossc', ..
         ',pmutationc,crowdfactorc,crowdsubpopc,nclassc,l', ..
         ',smultiple,last,',k,',it,l+4);'])
      end;
    end;
    if ~rem(it,dclassc);  
      for i=1:ntypes;  
        disp(' ')
        s='  Classifier System for Type %g Agents after Genetic Algorithm: \n';
        fprintf(s,i)
        disp('  --------------------------------------------------------------')
        disp(' ')
        eval(['disp([[1:nclassc]'',CSc',int2str(i),',[1:nclassc]''])'])
      end;
    end;
    if sum(initgac);
      for i=1:nsend;
        if initgac(send(i));
          k=int2str(send(i));
          disp(' ')
          s=['  Classifier System for Type %g Agents after Genetic', ..
              ' Algorithm: \n'];
          fprintf(s,send(i))
          disp(['  -------------------------------------------------',  ..
                   '-------------'])
          disp(' ')
          eval(['disp([[1:nclassc]'',CSc',k,',[1:nclassc]''])'])
          initgac(send(i))=0;
        end;
      end;
    end;
  end;
  if ~rem(it,dclasst) & runit(it)==0; 
    for i=1:ntypes;
      disp(' ')
      fprintf('  Trade Classifier System for Type %g Agents: \n',i)
      disp('  ----------------------------------------------')
      disp(' ')
      eval(['disp([[1:nclasst]'',CSt',int2str(i),',[1:nclasst]''])'])
    end;
  end;
  if ~rem(it,dclassc) & runit(it)==0;
    for i=1:ntypes;
      disp(' ')
      fprintf('  Consume Classifier System for Type %g Agents: \n',i)
      disp('  ------------------------------------------------')
      disp(' ')
      eval(['disp([[1:nclassc]'',CSc',int2str(i),',[1:nclassc]''])'])
    end;
  end;
end;
