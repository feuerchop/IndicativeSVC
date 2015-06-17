function c=marker_color2(i)
% MARKER_COLOR Returns marker color.
%
% Synopsis:
%  c=marker_color(i)
%
% See also MARKER_TYPE
%

% About: Statistical Pattern Recognition Toolbox
% (C) 1999-2003, Written by Vojtech Franc and Vaclav Hlavac
% Czech Technical University Prague
% Faculty of Electrical Engineering
% Center for Machine Perception

% Modifications:
%  7-jan-2003, VF, created

%COLORS=['r','b','y','m','c','k','g'];
COLORS=['b','g','y','m','c','r','k'];

c=COLORS(mod(i-1,length(COLORS))+1);

return;