using Flux
using Zygote
using Optimisers
using Plots
using CSV
using DataFrames
using Statistics
using LinearAlgebra
using StatsBase
using ITensors
using Combinatorics

include("RNNCause.jl");
include("Shadow.jl");
using .ShadowN1d
using .GateRF

N_sites = 9;L_layers = 4;g_gate = Int64((N_sites-1)/2);
p = ones(N_sites);
ShadowN1d.N_sites = N_sites;ShadowN1d.L_layers = L_layers;ShadowN1d.g_gate = g_gate;
ShadowN1d.ψ₀,ShadowN1d.s0  = ShadowN1d.ψInitial();
llind = N_sites; Ksp,lind = ShadowN1d.KsamFixL(llind);
ShadowN1d.ψₜₐᵣ = ShadowN1d.ψsupp(lind);
Tₜₒₜ = 2*L_layers;
nᵢ = length(ShadowN1d.U₂ₜ);nh1 = 320;nh2 = 320; nₗ = g_gate;
GateRF.Tₜₒₜ = Tₜₒₜ;GateRF.nᵢ = nᵢ;GateRF.nh1 = nh1;GateRF.nh2 = nh2;
GateRF.nₗ = nₗ;GateRF.nₛ = N_sites;

FgaR = GateRF.NNbuild();ep = 1;
llind = rand([2;4;6;8]); #llind = rand([3;5;7]);
kall = collect(combinations(collect(1:N_sites), llind));cap=5;
Smem = [];Wmem = [];Kmem = [];weₒ=[];
kall = collect(combinations(collect(1:N_sites), llind));Lmem = [];Limem = [];
Flux.reset!(FgaR);
Ksp,lind = ShadowN1d.KsamFixL(llind);
ShadowN1d.ψₜₐᵣ = ShadowN1d.ψsupp(lind);
w₀ = zeros(Float32,nᵢ,nₗ);w₀[1,:] .=1.;w₀ = reshape(w₀,nᵢ*nₗ,1);
Mw1,Mh1,b1,h1,Mw2,Mh2,b2,h2,Mo = GateRF.paraGet(FgaR);
W₀,W₀ind,dLdo,K0 = GateRF.Evolute(Ksp,w₀,Mw1,Mh1,b1,h1,Mw2,Mh2,b2,h2,Mo,ep);
W₁ = W₀ind[2:end];
Snorm = ShadowN1d.loss(N_sites,L_layers,g_gate,W₁,p);
push!(Smem,[copy(Snorm)]);push!(weₒ,copy(Snorm));
push!(Wmem,W₀);push!(Kmem,K0);push!(Lmem,Ksp);push!(Limem,lind);
epoch = 100000;rep = 10;
Loo = [];
rule = Optimisers.Adam(0.001,(0.9,0.999))  # use the Adam optimiser with its default settings
opt_step = Optimisers.setup(rule, FgaR);  # initialise this optimiser's momentum etc.
Acₒ = copy(W₁);OO = zeros(nᵢ,1);Gparaₒ = GateRF.paraGet(FgaR);
Wₜⱼ=[1.];Dθ = [1];dᵦ = 0.01;Lmin=[];
W₀r = [];W₁r= []; weir = [];Snor = [];dLdor = [];K₀r = [];K0c = [];dLdoc = [];

for k in 1:epoch
    Flux.reset!(FgaR);
    llind =  rand([2;4;6;8]);
    #llind = rand([3;5;7]);
    Ksp,lind = ShadowN1d.KsamFixL(llind);
    ldif = [sum(abs.(Lmem[k] - Ksp)) for k in 1:length(Lmem)];
    lm1,bl = findmin(ldif);

    if lm1>0.5
        wee = 2^N_sites;
    else
        wee = weₒ[bl];
    end

    ShadowN1d.ψₜₐᵣ = ShadowN1d.ψsupp(lind);
    W₀r = [];W₁r= []; weir = [];Snor = [];dLdor = [];
    w₀ = zeros(Float32,nᵢ,nₗ);w₀[1,:] .=1.;w₀ = reshape(w₀,nᵢ*nₗ,1);
    Mw1,Mh1,b1,h1,Mw2,Mh2,b2,h2,Mo = GateRF.paraGet(FgaR);
    W₀,W₀ind,dLdo,K0 = GateRF.Evolute(Ksp,w₀,Mw1,Mh1,b1,h1,Mw2,Mh2,b2,h2,Mo,ep);
    W₁ = W₀ind[2:end];push!(W₁r,W₀ind);
    nswap = count(x->x==2,W₁)/(lind[end]-lind[1]);
    Snorm = ShadowN1d.loss(N_sites,L_layers,g_gate,W₁,p);
    Snorm = Snorm+nswap*0.0005; push!(Snor,Snorm);
    ws = Snorm^(1/N_sites); 
    push!(weir,ws);
    Reward = sign(wee-Snorm+0.2)*0.001;
    W₀r =  copy(W₀);K₀r = copy(K0);
    dLdor = copy(dLdo*Reward);
    kr = 1;
    while kr < rep
        
        w₀ = zeros(Float32,nᵢ,nₗ);
        for wrr in 1:nₗ
            w₀[rand(1:nᵢ),wrr] =1.;
        end
        w₀ = reshape(w₀,nᵢ*nₗ,1);
        Mw1,Mh1,b1,h1,Mw2,Mh2,b2,h2,Mo = GateRF.paraGet(FgaR);
        W₀,W₀ind,dLdo,K0 = GateRF.Evolute(Ksp,w₀,Mw1,Mh1,b1,h1,Mw2,Mh2,b2,h2,Mo,ep);
        W₁ = W₀ind[2:end];push!(W₁r,W₀ind);
        nswap = count(x->x==2,W₁)/(lind[end]-lind[1]);
        Snorm = ShadowN1d.loss(N_sites,L_layers,g_gate,W₁,p);
        Snorm = Snorm+nswap*0.0005;push!(Snor,Snorm);
        ws = Snorm^(1/N_sites);
        Reward = sign(wee-Snorm+0.2)*0.001;
        dLdo = dLdo*Reward;
        W₀r = [[W₀r[i] W₀[i]] for i in 1:Tₜₒₜ+1];
        K₀r = [[K₀r[i] K0[i]] for i in 1:Tₜₒₜ+1];
        dLdor = [[dLdor[i] dLdo[i]] for i in 1:Tₜₒₜ];
        kr += 1;
    end
    
    wm = mean(Snor);
    wm1,bm = findmin(Snor);push!(Loo,mean(Snor.^(1/llind)));push!(Lmin,wm1^(1/llind));
    if minimum(ldif)>0.5
        bm1 = copy(bm);
        q1 = [W₀r[k][:,bm1] for k in 1:Tₜₒₜ+1];
        q2 = [K₀r[k][:,bm1] for k in 1:Tₜₒₜ+1];
        push!(Wmem,q1);push!(Kmem,q2);
        push!(Lmem,Ksp);push!(Limem,lind);
        push!(Smem,[copy(wm1)])
        weₒ = [weₒ;wm];

    else
        if weₒ[bl]>=wm
            weₒ[bl] = copy(wm);
        end
        if maximum(Smem[bl])+0.5 > wm1
            bm1 = copy(bm);
            q1 = [W₀r[k][:,bm1] for k in 1:Tₜₒₜ+1];
            q2 = [K₀r[k][:,bm1] for k in 1:Tₜₒₜ+1];
            Wmem[bl] = [[q1[k] Wmem[bl][k]] for k in 1:Tₜₒₜ+1];
            Kmem[bl] = [[q2[k] Kmem[bl][k]] for k in 1:Tₜₒₜ+1];
            Smem[bl] = [copy(wm1);Smem[bl]];
            if length(Smem[bl])>cap
                pp = sortperm(Smem[bl]);pp1 = pp[1:cap];
                Wmem[bl] = [Wmem[bl][k][:,pp1] for k in 1:Tₜₒₜ+1];
                Kmem[bl] = [Kmem[bl][k][:,pp1] for k in 1:Tₜₒₜ+1];
                Smem[bl] = Smem[bl][pp1];
            end
        end
    end
    

    Flux.reset!(FgaR);
    dθ = GateRF.paraGrad(FgaR,K₀r[1:end-1],dLdor);
    opt_step, FgaR = Optimisers.update(opt_step, FgaR, dθ);
    
    if mod(k,5000)==0
        println("epoch:",k,"   support:",lind,"    loss:",mean(Snor.^(1/llind)));
    end

    if mod(k,200)==0
        wwm = Wmem[1];kkm = Kmem[1];
        for kk in 2:length(Wmem)
            wwm = [[Wmem[kk][k] wwm[k]] for k in 1:Tₜₒₜ+1];
            kkm = [[Kmem[kk][k] kkm[k]] for k in 1:Tₜₒₜ+1]; 
        end
        for k2 in 1:200
            Flux.reset!(FgaR);
            dθ = GateRF.crossGrad(FgaR,kkm[1:length(kkm)-1],wwm[2:length(wwm)]);
            opt_step, FgaR = Optimisers.update(opt_step, FgaR, dθ);
        end
    end

end

# Gate-sequence generator prediction

Flux.reset!(FgaR);
Mw1,Mh1,b1,h1,Mw2,Mh2,b2,h2,Mo = GateRF.paraGet(FgaR);mww=[];wall=[];stww=[];
for kss in 2:N_sites
    kall = collect(combinations(collect(1:N_sites), kss));ww=[];
        for nk in 1:length(kall)
        lind = kall[nk];Ksp = zeros(Float32,N_sites,1);
        Ksp[lind] .=1;
        ShadowN1d.ψₜₐᵣ = ShadowN1d.ψsupp(lind);
        w₀ = zeros(Float32,nᵢ,nₗ);w₀[1,:] .=1.;w₀ = reshape(w₀,nᵢ*nₗ,1);
        W₀,W₀ind,dLdo,K0 = GateRF.Evolute(Ksp,w₀,Mw1,Mh1,b1,h1,Mw2,Mh2,b2,h2,Mo,0)
        W₁ = W₀ind[2:end];
        Snorm = ShadowN1d.loss(N_sites,L_layers,g_gate,W₁,p);
        ws = Snorm^(1/kss);
        ww = [ww;ws];
    end
    mww=[mww;mean(ww)];stww =[stww;std(ww)];wall=[wall;ww];
end
