function mostsimilar=crowding(child,CS,crowdingfactor,crowdingsubpop,M,l)
if crowdingfactor<1; crowdingfactor=1; end;
matchmax=-1; mostsimilar=0;
for j=1:crowdingfactor;
  worst=ceil(rand(crowdingsubpop,1)*M);
  worst=[worst,CS(worst,l+3)];
  worststr=find(worst(:,2)<=min(worst(:,2)));
  popmember=worst(worststr(1),1);
  match=sum(child(1:l)==CS(popmember,1:l));
  if match>matchmax;
    matchmax=match;
    mostsimilar=popmember;
  end;
end;
