function annoteBeta(model)
% annotate beta values for svc model
    hold on;
    for i = 1:length(model.sv.inx)
        text(model.sv.X(i,1),model.sv.X(i,2),['\leftarrow ',num2str(model.Alpha(i))]);
    end
end

