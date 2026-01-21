load [mcgrattan.genetic.hillclimb]intermed


while gen < maxgen;  
  gen=gen+1;
  j=1;
  while j<popsize;
    %  (a) spin roulette wheel to select 
    mate1=select(popsize,sumfitness,oldpopf);
    mate2=select(popsize,sumfitness,oldpopf);
    if rand<pcross;
      jcross=1+floor((lchrom-1)*rand);
      ncross=ncross+1;
    else;
      jcross=lchrom;
    end;
    newpopc(j,:)=[abs(oldpopc(mate1,1:jcross)-(rand(1,jcross)<pmutation)), ..
        abs(oldpopc(mate2,jcross+1:lchrom)-(rand(1,lchrom-jcross)<pmutation))];
    newpopc(j+1,:)=[abs(oldpopc(mate2,1:jcross)-(rand(1,jcross)<pmutation)), ..
        abs(oldpopc(mate1,jcross+1:lchrom)-(rand(1,lchrom-jcross)<pmutation))];
    newpopx(j:j+1,:)=decode(newpopc(j:j+1,:),nparms,lparm,maxparm,minparm);
    newpopo(j,1)=objfunc(newpopx(j,:));
    newpopo(j+1,1)=objfunc(newpopx(j+1,:));
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
        [fbest,gbest]=objfunc(xbest);
      else;
        dev=diag(sqrt(1e-15)*max(abs(xbest'),typsiz));   
        sdev=diag(side-1)*dev;
        for i=1:nparms;
          gbest(i,1)=(objfunc(xbest+dev(i,:))-objfunc(xbest-sdev(i,:)))/ ..
                     (side(i)*dev(i,i));
        end;
      end;
      disp(' Associated parameter and gradient vectors:')
      disp([xbest',gbest])
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
    fprintf('    maxf = %12.4e, minf = %12.4e, avgf= %12.4e\n',maxf,minf,avgf)
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
  if ~rem(gen,20);
    save intermed
  end;
end;
