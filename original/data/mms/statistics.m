function [maxf,minf,avg,sumfitness,best]=statistics(popf,popsize);
maxf=max(popf);
minf=min(popf);
sumfitness=sum(popf);
avg=sumfitness/popsize;
if nargout>4;
  best=find(popf>=max(popf)-1e-8);
  best=best(1);
end;
