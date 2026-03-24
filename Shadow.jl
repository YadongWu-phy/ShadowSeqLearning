module ShadowN1d

using ITensors
using Zygote
using LinearAlgebra

Mₛ = [1 1/√3;1/√3 1/3];Idnₛ = Matrix(1.0I,2,2);
Uᵢ₀ = Matrix(1.0I,4,4);
Uᵪ = [1. 0 0 0;
      0. 0 1 0;
      0. 1 0 0;
      0. 0 0 1];
Uₛ = [1. 0 0 0;
    0. 0 1/3 2/(3*√3);
    0. 1/3 0 2/(3*√3);
    0. 2/(3*√3) 2/(3*√3) 5/9];
Uₕ = [1. 0 0 0;
    0. 1/5 1/5 √3/5;
    0. 1/5 1/5 √3/5;
    0. √3/5 √3/5 3/5];
U₂ₜ = (Uᵢ₀,Uᵪ,Uₛ);
#U₂ₜ = (Uᵢ₀,Uₕ,Uₛ);

Base.@kwdef mutable struct Param
    N_sites :: Int64
    L_layers :: Int64
    g_gate :: Int64
end

function ψInitial()
    s0 = siteinds("Qubit",N_sites);
    ψ₀ = MPS(s0,"0");
    return ψ₀,s0
end

function KsamFixL(llind)
    k = zeros(Float32,N_sites,1);
    lp = sortperm(rand(N_sites));
    lind = lp[1:llind];
    k[lind] .=1;
    return k, sort(lind)
end



function Ksam()
    Ksub = sign.(rand(Float32,N_sites,1).-0.5);Ksub = (Ksub .+1)./2;
    lind = [];
    for lk in 1:N_sites
        if Ksub[lk]>0.1
            push!(lind,lk);
        end
    end
    k = zeros(Float32,N_sites,1);k[lind] .=1;
    return k,lind
end
    

function ψsupp(lind)
    wₚₐ = [([1 0;0 1.0/sqrt(3)],(k,)) for k in 1:N_sites];
    Wₚₐ = ops(wₚₐ,s0);
    aₖ = repeat(["0"],N_sites);aₖ[lind] .="1";
    ψₜ₀ = MPS(s0,aₖ);
    ψₜₐᵣ = apply(Wₚₐ,ψₜ₀;cutoff=1e-8);
    return ψₜₐᵣ
end

function two_qubit_EF(uo,ue)
    wₒ = [(U₂ₜ[uo[k]], (2*k-1,2*k)) for k in 1:g_gate];
    wₑ = [(U₂ₜ[ue[k]], (2*k,2*k+1)) for k in 1:g_gate];
    return [wₒ;wₑ];
end

function brick_wall(g_gate, L_layers, u)
    ran1 = 1:g_gate
    circuit = [];
    for k in 1:L_layers
        uo = u[ran1 .+ (2*k-2)*g_gate];ue = u[ran1 .+ (2*k-1)*g_gate];
        circuit = [circuit;two_qubit_EF(uo,ue)];         
    end
    return circuit
end

function single_qubit_mea(N_sites,p)
    pₘₑₐₛ = [((1-p[k])*Idnₛ + p[k]*Mₛ, (k,)) for k in 1:N_sites];
    return pₘₑₐₛ
end

function loss(N_sites,L_layers,g_gate,u,p)
    mᵪ = single_qubit_mea(N_sites,p);
    Uₘₑₐₛ = ops(mᵪ, s0);
    ψₘₑₐ = apply(Uₘₑₐₛ, ψ₀;cutoff = 1e-8);
    
    uᵪ = brick_wall(g_gate, L_layers, u);
    U_bri = ops(uᵪ, s0);
    ψ_bri = apply(U_bri, ψₜₐᵣ;cutoff = 1e-8);
    weight_A = inner(ψ_bri,ψₘₑₐ)+1e-8;
    lossa = 1 ./weight_A;
    return lossa
end





end
