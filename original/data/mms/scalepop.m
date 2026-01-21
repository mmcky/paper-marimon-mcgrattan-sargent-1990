function [popf,sumfitness]=scalepop(popo,maxf,minf,avgf,fmultiple)
newmin=1;
if avgf>0;
  if minf>(fmultiple*avgf-maxf+(maxf-avgf)*newmin/avgf)/(fmultiple-1);
    delta=maxf-avgf;
    a=(fmultiple-1)*avgf/delta;
    b=avgf*(maxf-fmultiple*avgf)/delta;
  else;
    delta=avgf-minf;
    a=(avgf-newmin)/delta;
    b=newmin-a*minf;
  end;
else;
  a=1;
  b=abs(minf)+newmin;
end;
%popf=a*popo+b;
popf=max(a*popo+b,ones(length(popo),1));
sumfitness=sum(popf);
