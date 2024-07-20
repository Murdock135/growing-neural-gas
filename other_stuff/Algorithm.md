# Neural gas algorithm

Algorithm:

1. Initialize:
    + inputs, $\{X\}_1^N = \{x_1, x_2, ...x_N\}$,
    + neurons, $\{U\}_1^K = \{u_1, u_2, ...u_K\}$, $K<N$ // "U" stands for "unit"
    + topology (connection matrix) $A=[\textbf{0}]_{K,K}$

    where $x_i, u_i \in \mathbb{R}^n$

2. Set maximum age for connections, $\tau$

3. Loop until maximum number of iterations M:
    1. Randomly select an input $x_i$
    2. Loop over neurons $\{U\}_1^K$:
        + Let current neuron be $u_c$
        + Find $|B_c|$ where $B_c=\{u_b \; | \; norm(u_b, x_i) < norm(u_c, x_i) \}$
    3. Sort neurons by distance to $x_i$, $\{U_{sorted}\} = \{u_{i_0}, u_{i_1}... u_{i_{K-1}}\}$
    4. Update/replace all neurons with,
        $$
        u_c = u_c + \epsilon.e^{- |B_c|/ \lambda } (x_i - u_c)
        $$
        for c={1,2,...K}.

    5. Make connections between closest neurons ($u_{i_0}, u_{i_1}$) if connection doesn't exist already else increase connection age, $C_{u_{i_0}, u_{i_1}} = C_{u_{i_0}, u_{i_1}}+1$
    6. if $C_{u_{i_0}, u_{i_1}}>\tau$, set $C_{u_{i_0}, u_{i_1}}=0$

In the original paper, the parameters $\lambda, \epsilon, \tau$ were defined by a function of time (iteration),
$g(t)=g_i \frac{g_f}{g_i}^{\frac{t}{t_{max}}}$ where,
    $\lambda_i=30, \lambda_f=0.01, \epsilon_i=0.3, \epsilon_f=0.05, \tau_i=20, \tau_f=200, t_{max}=40000$

Example algorithm flowchart:

Initial 'distances' array:

            data = np.array([[1, 2],
                              [3, 4],
                              [5, 6]])
                        Shape: (3, 2)

                        code_vectors = np.array([[0, 0],
                                                  [1, 1]])
                        Shape: (2, 2)

                             ┌───────────────┐
                             │  data - cv    │
                             └───────┬───────┘
                                     │
            ┌───────────────────────┴───────────────────────┐
            │[[ 1  2]      [[ 0  1]                          │
            │ [ 3  4]       [ 2  3]                          │
            │ [ 5  6]]      [ 4  5]]                         │
            └───────────────────────┬───────────────────────┘
                                     │
                             ┌───────┴───────┐
                             │np.linalg.norm(│
                             │   axis=1)     │
                             └───────┬───────┘
                                     │
            ┌───────────────────────┴───────────────────────┐
            │[2.23606798, 5.        , 7.81024968]            │
            │[1.41421356, 3.60555128, 6.40312424]            │
            └───────────────────────┬───────────────────────┘
                                     │
                             ┌───────┴───────┐
                             │    np.array   │
                             └───────┬───────┘
                                     │
            ┌───────────────────────┴───────────────────────┐
            │[[2.23606798, 5.        , 7.81024968],          │
            │ [1.41421356, 3.60555128, 6.40312424]]          │
            └───────────────────────┬───────────────────────┘
                                     │
                             ┌───────┴───────┐
                             │       .T      │
                             └───────┬───────┘
                                     │
            ┌───────────────────────┴───────────────────────┐
            │[[2.23606798, 1.41421356],                      │
            │ [5.        , 3.60555128],                      │
            │ [7.81024968, 6.40312424]]                      │
            └───────────────────────┬───────────────────────┘
                                     │
                                Final 'distances'
                                Shape: (3, 2)
            
            ┌───────────────────────┐
            │[[2.23606798, 1.41421356], 
            │ [5.        , 3.60555128],
            │ [7.81024968, 6.40312424]]
            └───────────────────────┘
                        │
                        ▼
            ┌───────────────────────┐
            │np.argsort(distances,  │
            │           axis=1)     │
            └───────────────────────┘
                        │
                        ▼
            ┌───────────────────────┐
            │[[1, 0],                
            │ [1, 0],
            │ [1, 0]]
            └───────────────────────┘
                        │
                        ▼
                  'sorted_indices'

                        │
                  ┌─────┴─────┐
                  │ i, cv in  │  
                  │enumerate(code_vectors)
                  └─────┬─────┘
                        │
                  ┌─────┴─────┐
                  │  i = 0    │
                  │  cv = [0, 0]
                  └─────┬─────┘
                        │
                        ▼
            ┌───────────────────────┐
            │ranks = np.zeros_like( │
            │  sorted_indices[:, 0])│  
            └───────────────────────┘
                        │
                        ▼
            ┌───────────────────────┐
            │ ranks = [0, 0, 0]     │
            └───────────────────────┘
                        │
                  ┌─────┴─────┐
                  │   j in    │
                  │range(n_samples)
                  └─────┬─────┘
                        │
                  ┌─────┴─────┐
                  │   j = 0   │
                  └─────┬─────┘
                        │
                        ▼
            ┌───────────────────────┐
            │ranks[0] = np.where(   │
            │      sorted_indices   │
            │         [0] == 0)[0]  │
            └───────────────────────┘
                        │
                        ▼
            ┌───────────────────────┐
            │ ranks = [1, 0, 0]     │
            └───────────────────────┘

                  ┌─────┴─────┐
                  │   j = 1   │  
                  └─────┬─────┘
                        │
                        ▼
            ┌───────────────────────┐
            │ranks[1] = np.where(   │
            │      sorted_indices   │
            │         [1] == 0)[0]  │
            └───────────────────────┘
                        │
                        ▼
            ┌───────────────────────┐
            │ ranks = [1, 1, 0]     │
            └───────────────────────┘

                  ┌─────┴─────┐
                  │   j = 2   │
                  └─────┬─────┘
                        │
                        ▼
            ┌───────────────────────┐
            │ranks[2] = np.where(   │
            │      sorted_indices   │
            │         [2] == 0)[0]  │
            └───────────────────────┘
                        │
                        ▼
            ┌───────────────────────┐
            │ ranks = [1, 1, 1]     │
            └───────────────────────┘
