## Quantum angular momenta
Following functions are included in the codes `quantang.py`.
* The reduced rotational matrix 
`Rd(j,m,n,t,d=0)` is designed to calculate the Wigner D-matrix. 
`d` in `Rd(j,m,n,t,d=0)` denotes the order-$d$ derivative. 

* Spherical harmonics 
`SHY_list(l,m,t,phi)`,
`l`,`m` angular momentum and its projection on $z-$axis.
`t`,`phi` for $\theta$ and $\phi$.

* CG coefficients 
`CGcoin(l,s,j,mj)` will return the CG coefficients for given $l$, $s$, $j$, $m_j$.
