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

include("EFShadow.jl");
using .EFshadow

N_sites = 5;L_layers = 4;gradtol = 1e-8;
p = ones(N_sites);
EFshadow.N_sites = N_sites;EFshadow.L_layers = L_layers;
ψ₀,s0  = EFshadow.ψInitial();EFshadow.ψ₀ = ψ₀;EFshadow.s0 = s0;
lind = collect(1:N_sites);k_len = length(lind);
ψₜₐᵣ = EFshadow.ψsupp(lind);EFshadow.ψₜₐᵣ = ψₜₐᵣ;
Uₚ₁ = EFshadow.Uₚ₁;Uₚ₂ = EFshadow.Uₚ₂;
p₀ = ones(N_sites);d=2;rep = 10;epoch = 10000;lr = 0.1;
LOSS = [];SAC = []; SAD = [];
for rp in 1:rep
    xₜ₀ = 1*randn((N_sites-1)*L_layers);
    xᵦ₀ = 1*randn((N_sites-1)*L_layers);
    xₚ₀ = 1*randn(N_sites);
    Loss =[];l1 = 1;
    s_xₜ = 0 * copy(xₜ₀);t_xₜ = 0 * copy(xₜ₀);
    s_xᵦ = 0 * copy(xᵦ₀);t_xᵦ = 0 * copy(xᵦ₀);
    s_xₚ = 0 * copy(xₚ₀);t_xₚ = 0 * copy(xₚ₀);
    αₘ₁ = [];αₘ₂ = []; pₘᵢₙ = [];wₘᵢₙ = 5;

    for k in 1:epoch
        t,dxt = EFshadow.paraX(xₜ₀);β,dxβ = EFshadow.paraX(xᵦ₀);
        W_AC,W_AD,dtW_AC,dtW_AD,dβW_AC,dβW_AD = EFshadow.entro_bound(t,β);            
        loss0, U0_bri,U0_mea, ψ0_bri,ψ0_mea,weight0_A = EFshadow.loss(N_sites,L_layers,W_AC,W_AD,p₀);
        xₜ = copy(xₜ₀);xᵦ= copy(xᵦ₀);xₚ = copy(xₚ₀);
        w₁ = loss0^(1/k_len);
        if wₘᵢₙ > w₁
            αₘ₁ = copy(W_AC);αₘ₂ = copy(W_AD); pₘᵢₙ = copy(p₀);
            wₘᵢₙ = w₁;
        end
        
        for nl in 1:L_layers
            ino = 0;
            for nn in 1:2:N_sites-1
                no = Int32((nn+1)/2);
                dU_o1 = copy(U0_bri);
                dU_o1[no + (nl-1)*(N_sites-1)] = op((Uₚ₁,nn,nn+1),s0);
                dψo1 = apply(dU_o1,ψₜₐᵣ;cutoff=1e-8);
                dwo1 = inner(ψ0_mea, dψo1);
                dlo1 = -1/weight0_A^(-2) * dwo1;
                #dlo1 = -1/k_len*weight0_A^(-1-1/k_len) * dwo1;
                #dlo1 = -dwo1;
                
                dU_o2 = copy(U0_bri);
                dU_o2[no + (nl-1)*(N_sites-1)] = op((Uₚ₂,nn,nn+1),s0);
                dψo2 = apply(dU_o2,ψₜₐᵣ;cutoff=1e-8);
                dwo2 = inner(ψ0_mea, dψo2);
                dlo2 = -1/weight0_A^(-2) * dwo2;
                #dlo2 = -1/k_len*weight0_A^(-1-1/k_len) * dwo2;
                #dlo2 = -dwo2;
                
                a_in = nn+(nl-1)*(N_sites-1);
                dxtW = dxt[a_in]*(dlo1* dtW_AC[a_in]+dlo2*dtW_AD[a_in]);
                dla1,s_xₜ[a_in],t_xₜ[a_in] = EFshadow.Adams_Optim(dxtW,s_xₜ[a_in],t_xₜ[a_in],k);
                #dla1 = dxtW;
                xₜ[a_in] -= dla1 * lr;
                dxβW = dxβ[a_in]*(dlo1*dβW_AC[a_in]+dlo2*dβW_AD[a_in]);
                dla2,s_xᵦ[a_in],t_xᵦ[a_in] = EFshadow.Adams_Optim(dxβW,s_xᵦ[a_in],t_xᵦ[a_in],k);
                #dla2 = dxβW;
                xᵦ[a_in] -= dla2 * lr;
                ino +=1;
            end
            
            for nn in 2:2:N_sites-1
                ne = Int32(nn/2+ino);
                dU_e1 = copy(U0_bri);
                dU_e1[ne + (nl-1)*(N_sites-1)] = op((Uₚ₁,nn,nn+1),s0);
                dψe1 = apply(dU_e1,ψₜₐᵣ;cutoff=1e-8);
                dwe1 = inner(ψ0_mea,dψe1);
                dle1 = -1/weight0_A^(-2) * dwe1;
                #dle1 = -1/k_len*weight0_A^(-1-1/k_len) * dwe1;
                #dle1 = -dwe1;
                
                dU_e2 = copy(U0_bri);
                dU_e2[ne + (nl-1)*(N_sites-1)] = op((Uₚ₂,nn,nn+1),s0);
                dψe2 = apply(dU_e2,ψₜₐᵣ;cutoff=1e-8);
                dwe2 = inner(ψ0_mea, dψe2);
                dle2 = -1/weight0_A^(-2) * dwe2;
                #dle2 = -1/k_len*weight0_A^(-1-1/k_len) * dwe2;
                #dle2 = -dwe2;
                
                a_in = nn+(nl-1)*(N_sites-1);
                dxtW = dxt[a_in]*(dle1* dtW_AC[a_in]+dle2*dtW_AD[a_in]);
                dla1,s_xₜ[a_in],t_xₜ[a_in] = EFshadow.Adams_Optim(dxtW,s_xₜ[a_in],t_xₜ[a_in],k);
                #dla1 = dxtW;
                xₜ[a_in] -= dla1 * lr;
                dxβW = dxβ[a_in]*(dle1*dβW_AC[a_in]+dle2*dβW_AD[a_in]);
                dla2,s_xᵦ[a_in],t_xᵦ[a_in] = EFshadow.Adams_Optim(dxβW,s_xᵦ[a_in],t_xᵦ[a_in],k);
                #dla2 = dxβW;
                xᵦ[a_in] -= dla2 * lr;
                
            end
        end
        
        l1 = loss0;
        Loss=[Loss;loss0];
        xₜ₀ = copy(xₜ);xᵦ₀ = copy(xᵦ);xₚ₀ = copy(xₚ);
        
    end
    LOSS=[LOSS;copy(Loss)];SAC=[SAC;copy(αₘ₁)];SAD=[SAD;copy(αₘ₂)];
    println("rep: ",rp,"  loss=",wₘᵢₙ)
end