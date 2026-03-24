module GateRF

using Flux
using Flux: gradient
using Flux:params
using Optimisers
using Zygote
using Random
using LinearAlgebra
using StatsBase

Base.@kwdef mutable struct Param
    nh1 :: Int64
    nh2 :: Int64
    nᵢ :: Int64
    nₗ :: Int64
    Tₜₒₜ:: Int64
    nₛ :: Int64
end

function doubleRNN(w1,Mw1,Mh1,b1,h1,Mw2,Mh2,b2,h2,Mo)
    hr1 = sigmoid(Mh1*h1+Mw1*w1+b1);
    hr2 = sigmoid(Mh2*h2+Mw2*hr1+b2);
    o₁ = Mo * hr2;
    p₁ = proOne(o₁);
    return hr1,hr2,o₁,p₁
end

function proOne(o₁)
    o₂ = reshape(o₁,nᵢ,nₗ);
    p₁ = softmax(o₂);
    return p₁
end

function NNbuild()
    r = Chain(RNN(nᵢ*nₗ + nₛ =>nh1,sigmoid),RNN(nh1 => nh2,sigmoid),Dense(nh2=> nᵢ*nₗ,bias = false));
    return r
end

function paraGet(r)
    g = Flux.params(r);
    Mw1,Mh1,b1,h1,Mw2,Mh2,b2,h2,Mo = [g[k] for k in 1:length(g)];
    return Mw1,Mh1,b1,h1,Mw2,Mh2,b2,h2,Mo
end

function paraGrad(c,a,d) 
    g,_ = gradient(c, a) do m, x  # calculate the gradients
        b1 = [m(x2) for x2 in x];
        b2 = [tr(d[k]'*b1[k]) for k in 1:Tₜₒₜ];
        mean(b2)
        end;
    return g
end

function crossGrad(c,a,ac)
    g,_ = gradient(c,a) do m,x
        b1 = [m(x2) for x2 in x];
        b2 = [min.(softmax(x3),1) for x3 in b1];
        #b2 = [softmax(x3) for x3 in b1];
        b3 =  [Flux.Losses.crossentropy(b2[k].*nₗ,ac[k]) for k in 1:length(b2)];
        #b3 =  [Flux.Losses.kldivergence(b2[k],ac[k]./nₗ) for k in 1:length(b2)];
        mean(b3)
    end
    return g
end

function maxSample(o₁)
    _,b = findmax(o₁);
    w₁ = zeros(Float32,nᵢ,1);w₁[b[1],1] = 1.;
    return w₁,b[1]
end

function weightSample(o₁)
    b = sample(1:nᵢ, Weights(o₁[:]));
    w₁ = zeros(Float32,nᵢ,1);w₁[b[1],1] = 1.;
    return w₁,b[1]
end

function Evolute(k0,w₀,Mw1,Mh1,bb1,h1,Mw2,Mh2,bb2,h2,Mo,ep)
    W₀ = [copy(w₀)];ks = [k0;copy(w₀)];K0 = [copy(ks)];
    dLdo=[];
    W₀ind = 1;
    for k in 1:Tₜₒₜ
        hr1,hr2,_,p₁ = doubleRNN(ks,Mw1,Mh1,bb1,h1,Mw2,Mh2,bb2,h2,Mo);
        μ = rand();pp = p₁;
        if μ<ep
            ac,b1 = weightSample(pp[:,1]);W₀ind = [W₀ind;copy(b1)];
            dl1 = zeros(Float32,nᵢ);dl1[b1] = pp[b1,1] -1.;
            for kk in 2:nₗ
                a2,b2 = weightSample(pp[:,kk]);ac = [ac;a2];
                dl2 = zeros(Float32,nᵢ);dl2[b2] = pp[b2,kk] -1.;
                dl1 = [dl1;dl2];W₀ind = [W₀ind;copy(b2)];
            end
        else
            ac,b1 = maxSample(pp[:,1]);W₀ind = [W₀ind;copy(b1)];
            dl1 = zeros(Float32,nᵢ);dl1[b1] = pp[b1,1] -1.;
            for kk in 2:nₗ
                a2,b2 = maxSample(pp[:,kk]);ac = [ac;a2];
                dl2 = zeros(Float32,nᵢ);dl2[b2] = pp[b2,kk] -1.;
                dl1 = [dl1;dl2];W₀ind = [W₀ind;copy(b2)];
            end
        end
        push!(dLdo,dl1);
        #dLdo[:,k] = p₁ .-1.;
        h1 = copy(hr1);h2 = copy(hr2);
        w₀ = copy(ac);ks = [k0;w₀];
        W₀ = push!(W₀,w₀);K0 = push!(K0,ks);
    end
    return W₀,W₀ind,dLdo,K0
end






end
