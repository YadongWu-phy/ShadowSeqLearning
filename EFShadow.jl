module EFshadow
using ITensors
using Zygote
using LinearAlgebra

Base.@kwdef mutable struct Param
    N_sites :: Int64
    L_layers :: Int64
end

η₀ = [1.;0];
W₁ = [1 1/2;1/2 1];Wᵢₙᵥ₁=[4/3 -2/3;-2/3 4/3];
Mₛ = [1 1/√3;1/√3 1/3];Idnₛ = Matrix(1.0I,2,2);
dMₛ = Mₛ - Idnₛ;

d=2;
U₂₀ = [d^4 d^3 d^3 d^2;
       d^3 0 0 d^3;
       d^3 0 0 d^3;
       d^2 d^3 d^3 d^4]/d^4;
Uₐ₁ = zeros(4,4);Uₐ₁[2,2]=1;Uₐ₁[3,3]=1;
Uₐ₂ = zeros(4,4);Uₐ₂[2,3]=1;Uₐ₂[3,2]=1;
Uᵢₙᵥ = kron(Wᵢₙᵥ₁,Wᵢₙᵥ₁);
Pᵪ = [1 0.5 0.5 0.25;
      0 sqrt(3)/2 0 sqrt(3)/4;
      0 0 sqrt(3)/2 sqrt(3)/4;
      0 0 0 3/4];
Uₚ₀ = Pᵪ * Uᵢₙᵥ * U₂₀ * Uᵢₙᵥ * Pᵪ';
Uₚ₁ = Pᵪ * Uᵢₙᵥ * Uₐ₁ * Uᵢₙᵥ * Pᵪ';
Uₚ₂ = Pᵪ * Uᵢₙᵥ * Uₐ₂ * Uᵢₙᵥ * Pᵪ';

function ψInitial()
    s0 = siteinds("Qubit",N_sites);
    ψ₀ = MPS(s0,"0");
    return ψ₀,s0
end

function ψsupp(lind)
    wₚₐ = [([1 0;0 1.0/sqrt(3)],(k,)) for k in 1:N_sites];
    Wₚₐ = ops(wₚₐ,s0);
    aₖ = repeat(["0"],N_sites);aₖ[lind] .="1";
    ψₜ₀ = MPS(s0,aₖ);
    ψₜₐᵣ = apply(Wₚₐ,ψₜ₀;cutoff=1e-8);
    return ψₜₐᵣ
end


function entro_bound(t,β)
    n = 2.;
    tₗ = t * π; tᵤ = t * π /4;
    l_AC = 8^(-n)*(3*(1 .-cos.(tₗ)).^n+(5 .+3*cos.(tₗ)).^n);
    l_AD = 8^(-n)*(3*(1 .+cos.(tₗ)).^n+(5 .-3*cos.(tₗ)).^n);
    u_AC = 2^(1-n)*(cos.(tᵤ).^(2*n)+sin.(tᵤ).^(2*n));
    u_AD = 2^(1-2*n)*((1 .-sin.(2*tᵤ)).^n+(1 .+sin.(2*tᵤ)).^n);
    W_AC = β .* l_AC + (1 .-β) .* u_AC;
    W_AD = β .* l_AD + (1 .-β) .* u_AD;
    
    dtl_AC = 8^(-n)*(3*π*n*(1 .-cos.(tₗ)).^(n-1).*sin.(tₗ)
        -3*π*n*(5 .+3*cos.(tₗ)).^(n-1).*sin.(tₗ));
    dtl_AD = 8^(-n)*(3*π*n*(5 .-3*cos.(tₗ)).^(n-1).*sin.(tₗ)
        -3*π*n*(1 .+cos.(tₗ)).^(n-1).*sin.(tₗ));
    dtu_AC = 2^(1-n)*(-2*π/4*n*cos.(tᵤ).^(2*n-1).*sin.(tᵤ)
        +2*π/4*n*cos.(tᵤ).*sin.(tᵤ).^(2*n-1));
    dtu_AD = 2^(1-2*n)*(-2*π/4*n*cos.(2*tᵤ).*(1 .-sin.(2*tᵤ)).^(n-1)
        +2*π/4*n*cos.(2*tᵤ).*(1 .+sin.(2*tᵤ)).^(n-1));
    
    dtW_AC = β .* dtl_AC + (1 .-β).*dtu_AC;
    dtW_AD = β .* dtl_AD + (1 .-β).*dtu_AD;
    dβW_AC = l_AC-u_AC;
    dβW_AD = l_AD-u_AD;

    
    return W_AC,W_AD,dtW_AC,dtW_AD,dβW_AC,dβW_AD
end

function paraX(x)
    t = 1 ./(exp.(-x).+1);
    dxt = t - t.^2;
    return t,dxt
end

function Adams_Optim(dl,s,r,t)
    ϵ = 1;
    ρ₁ = 0.9;ρ₂ = 0.999;
    del = 1e-8;
    s1 = ρ₁ * s +(1-ρ₁)*dl;
    r1 = ρ₂ * r +(1-ρ₂)*dl^2;
    s = s1/(1-ρ₁^t); r = r1/(1-ρ₂^t);
    dl2 = ϵ * s / (r^0.5+del);
    return dl2,s1,r1
end

function two_qubit_EF(N_sites,αₐ₁,αₐ₂)
    wₒ = [(Uₚ₀+αₐ₁[k]*Uₚ₁+αₐ₂[k]*Uₚ₂, (k,k+1)) for k in 1:2:N_sites-1];
    wₑ = [(Uₚ₀+αₐ₁[k]*Uₚ₁+αₐ₂[k]*Uₚ₂, (k,k+1)) for k in 2:2:N_sites-1];
    return [wₒ;wₑ];
end

function brick_wall(N_sites, L_layers, αₐ₁,αₐ₂)
    ran1 = 1:N_sites-1
    circuit = [];
    for k in 1:L_layers
        circuit = [circuit;
                   two_qubit_EF(N_sites,αₐ₁[ran1 .+ (k-1)*(N_sites-1)],αₐ₂[ran1 .+ (k-1) * (N_sites-1)])];         
    end
    return circuit
end

function single_qubit_mea(N_sites,p)
    pₘₑₐₛ = [((1-p[k])*Idnₛ + p[k]*Mₛ, (k,)) for k in 1:N_sites];
    return pₘₑₐₛ
end

function loss(N_sites,L_layers,αₐ₁,αₐ₂,p)
    mᵪ = single_qubit_mea(N_sites,p);
    Uₘₑₐₛ = ops(mᵪ, s0);
    ψₘₑₐ = apply(Uₘₑₐₛ, ψ₀;cutoff = 1e-8);
    
    uᵪ = brick_wall(N_sites, L_layers, αₐ₁,αₐ₂);
    U_bri = ops(uᵪ, s0);
    ψ_bri = apply(U_bri, ψₜₐᵣ;cutoff = 1e-8);
    weight_A = inner(ψₘₑₐ, ψ_bri);
    lossa = 1 ./weight_A;
    return lossa,U_bri,Uₘₑₐₛ,ψ_bri,ψₘₑₐ,weight_A
end





end