function [a,b]=scalestr(maxs,mins,avgs,smultiple)
if avgs>0;
  if mins>(smultiple*avgs-maxs)/(smultiple-1);
    delta=maxs-avgs;
    a=(smultiple-1)*avgs/delta;
    b=avgs*(maxs-smultiple*avgs)/delta;
  else;
    delta=avgs-mins;
    a=avgs/delta;
    b=-mins*avgs/delta;
  end;
else;
  a=1;
  b=abs(mins);
end;
