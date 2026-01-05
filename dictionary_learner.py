import numpy as np
from itertools import combinations
from collections import defaultdict


def KL(p, q):
    mask = p > 0
    return np.sum(p[mask] * np.log(p[mask] / q[mask]))


# ============================================================
#               Entropy cache (for J_S)
# ============================================================

class EntropyCache:
    """
    Caches entropies H(S) for all subsets encountered.
    """
    def __init__(self, X):
        self.X = X
        self.cache = {}

    def H(self, S):
        if S not in self.cache:
            vals, counts = np.unique(self.X[:, S], axis=0, return_counts=True)
            p = counts / counts.sum()
            self.cache[S] = -np.sum(p * np.log(p + 1e-12))
        return self.cache[S]


# ============================================================
#           Support cache (exact frequency pruning)
# ============================================================

class SupportCache:
    """
    Exact supports with downward-closure pruning.
    """
    def __init__(self, X):
        self.X = X
        self.cache = {}

    def support(self, S):
        if S not in self.cache:
            self.cache[S] = np.mean(np.prod(self.X[:, S], axis=1))
        return self.cache[S]

# ============================================================
# ============================================================
#        Lazy marginal indexing (for projections)
# ============================================================

class LazyMarginals:
    """
    Lazily computes and caches marginal indices.
    """
    def __init__(self, states):
        self.states = states
        self.cache = {}

    def get(self, S):
        if S not in self.cache:
            idx = np.array([tuple(s[i] for i in S) for s in self.states])
            _, inv = np.unique(idx, axis=0, return_inverse=True)
            self.cache[S] = inv
        return self.cache[S]


# ============================================================
#           Incremental IPF (exact-mode only)
# ============================================================

def project_incremental(
    states,
    p,
    S,
    lazy_marginals,
    q_init,
    n_sweeps=3
):
    """
    Warm-started projection adding a single interaction S.
    """
    q = q_init.copy()
    idx = lazy_marginals.get(S)

    for _ in range(n_sweeps):
        p_m = np.bincount(idx, weights=p)
        q_m = np.bincount(idx, weights=q)

        ratio = np.ones_like(p_m)
        mask = q_m > 0
        ratio[mask] = p_m[mask] / q_m[mask]

        q *= ratio[idx]
        q /= q.sum()

    return q


# ============================================================
#                   Hierarchical layer
# ============================================================

class Layer:
    """
    One hierarchical layer with caching and two RI modes.

    mode:
        - "exact": KL-based refined information (projection geometry)
        - "proxy": |J_S| heuristic (Enouen-style)
    """

    def __init__(
        self,
        k,
        sigma,
        tau,
        prev_layer=None,
        d=None,
        mode="exact"
    ):
        assert mode in {"exact", "proxy"}

        self.k = k
        self.sigma = sigma
        self.tau = tau
        self.prev_layer = prev_layer
        self.d = d if d is not None else prev_layer.d
        self.mode = mode

        self.atoms = []
        self.importances = {}
        self.I = 0.0
        self.q = None

    # --------------------------------------------------------

    def build(
        self,
        X,
        states,
        p_emp,
        lazy_marginals,
        entropy_cache,
        support_cache
    ):
        # ---- previous dictionary and model
        if self.prev_layer is None:
            D_prev = []
            q_prev = np.ones_like(p_emp) / len(p_emp)
        else:
            D_prev = list(self.prev_layer.atoms)
            q_prev = self.prev_layer.q

        # ---- candidate generation
        candidates = self._generate_candidates(D_prev)

        # ---- frequency pruning
        candidates = [
            S for S in candidates
            if support_cache.support(S) >= self.sigma
        ]
        if not candidates:
            self.q = q_prev
            return self

        # ---- J_S proxy (cached entropies)
        J = {S: self._J_score(S, entropy_cache) for S in candidates}

        screened = [S for S in candidates if abs(J[S]) >= self.tau]
        if not screened:
            self.q = q_prev
            return self

        screened.sort(key=lambda S: abs(J[S]), reverse=True)

        # ---- sequential selection
        current_atoms = []
        current_q = q_prev

        for S in screened:

            if self.mode == "proxy":
                RI = abs(J[S])

            else:  # exact mode
                q_trial = project_incremental(
                    states, p_emp, S,
                    lazy_marginals,
                    current_q
                )
                RI = KL(q_trial, current_q)
            print("atom",S,"imp=",RI)
            if RI < self.tau:
                break

            current_atoms.append(S)
            self.importances[S] = RI
            self.I += RI

            if self.mode == "exact":
                current_q = q_trial

        self.atoms = current_atoms
        self.q = current_q
        return self

    # --------------------------------------------------------

    def _generate_candidates(self, prev_atoms):
        if self.k == 1:
            return [(i,) for i in range(self.d)]

        buckets = defaultdict(list)
        for A in prev_atoms:
            for T in combinations(A, self.k - 2):
                buckets[T].append(A)

        C = set()
        for group in buckets.values():
            for A, B in combinations(group, 2):
                S = tuple(sorted(set(A) | set(B)))
                if len(S) == self.k:
                    C.add(S)
        return list(C)

    # --------------------------------------------------------

    def _J_score(self, S, entropy_cache):
        k = len(S)
        J = 0.0
        for r in range(1, k + 1):
            for T in combinations(S, r):
                J += (-1)**(k - r) * entropy_cache.H(T)
        return J




if __name__ =="__main__":

    # example on EFTrip just change X for another example
    import itertools
    def generate_binary_pairwise_model(N=5000, 
                                    th1=0.1, th2=0.2, th3=-0.1,
                                    th12=1.0, th23=0.6):
       
      
        states = list(itertools.product([0,1], repeat=3))
        logp = []

        for x1, x2, x3 in states:
            lp = (th1*x1 + th2*x2 + th3*x3
                + th12*(x1*x2)
                + th23*(x2*x3))
            logp.append(lp)

        logp = np.array(logp)
        p = np.exp(logp - logp.max())
        p /= p.sum()

        idx = np.random.choice(len(states), size=N, p=p)
        X = np.array([states[i] for i in idx])
        return X


    X=generate_binary_pairwise_model()

    states, counts = np.unique(X, axis=0, return_counts=True)
    p_emp = counts / counts.sum()
    d = X.shape[1]

    layer_params = {
        1: dict(sigma=0, tau=0.0,),
        2: dict(sigma=1e-1, tau=2e-3),
        3: dict(sigma=1e-12, tau=0,),
    }

    lazy_marginals = LazyMarginals(states)
    entropy_cache  = EntropyCache(X)
    support_cache  = SupportCache(X)

    build_kwargs = dict(
        X=X,
        states=states,
        p_emp=p_emp,
        lazy_marginals=lazy_marginals,
        entropy_cache=entropy_cache,
        support_cache=support_cache,
    )

    Layers = {}
    prev_layer = None
    d = X.shape[1]

    # cumulative dictionary
    D_cum = []

    for k in range(1, 4):

        params = layer_params[k]

        layer = Layer(
            k=k,
            sigma=params["sigma"],
            tau=params["tau"],
            prev_layer=prev_layer,
            d=d if k == 1 else None,
            mode="exact"
        ).build(**build_kwargs)

        Layers[k] = layer
        prev_layer = layer

        # --------------------------------------------
        # accumulate atoms up to this layer
        D_cum.extend(layer.atoms)

        

        print(f"k={layer.k} | atoms={layer.atoms} | I_k={layer.I:.6f} | KL={KL(p_emp, layer.q):.6f}")


        

        # Structural early stopping
        if len(layer.atoms) == 0:
            print(f"Early stopping at k={k}")
            break



# █████████████████████████████████████████████████████████
# █████████████████████████████████████████████████████████
# █████████████████████████████████████████████████████████
# ██████████████████#:. :+████████████*:  :*███████████████
# ████████████████.#██████▓.%██████▓.%██████▒.▒████████████
# ██████████████▓.██████████+..  .- ██████████-*███████████
# ████████@:+.█+.▒███████████.████.████████████.+██████████
# ████████ █████ %██████████▓.████ ▒██████████▓.███████████
# ████████#██████.#████████▓.██████.@████████▓.████████████
# ███████████▒████▓. ▒██▓-.%████████▓..▓██▓:.@█████████████
# ██████████▒#█████████████████████████████████████████████
# ████████████████+███████████████████████████▒▒███████████
# ████████████████ =██████-     #     :██████▓ ████████████
# ████████████████=  ███-                ▓██-  ████████████
# ████████████████▓           ▓████           *████████████
# █████████████████.    :██████   ▓█████@     █████████████
# ██████████████████     @███@     *████     ▒█████████████
# ███████████████████                       ▒██████████████
# ████████████████████                     ▓███████████████
# █████████████████████#                 -█████████████████
# ███████████████████████▒             *███████████████████
# ████████████████████████████*   +████████████████████████
# █████████████████████████████████████████████████████████
# █████████████████████████████████████████████████████████
# █████████████████████████████████████████████████████████
# █████████████████████████████████████████████████████████
# █████████████████████████████████████████████████████████
# █████████████████████████████████████████████████████████
# █████████████████████████████████████████████████████████
