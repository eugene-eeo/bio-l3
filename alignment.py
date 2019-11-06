import numpy as np
from itertools import chain
from collections import defaultdict


def make_scoring_dict(alphabet, matrix):
    M = {}
    for i, a in chain(enumerate(alphabet), [(-1, None)]):
        for j, b in chain(enumerate(alphabet), [(-1, None)]):
            M[a, b] = matrix[i][j]
    return M


def dynprog(alphabet, scores, s, t):
    m = len(s) + 1
    n = len(t) + 1
    V = np.zeros((m, n), dtype=int)
    M = [[' ']*n for _ in range(m)]
    S = make_scoring_dict(alphabet, scores)

    for i in range(1, m):
        V[i][0], M[i][0] = max(
            (V[i-1][0] + S[s[i-1], None], 'U'),
            (0, 'T'),
        )

    for j in range(1, n):
        V[0][j], M[0][j] = max(
            (V[0][j-1] + S[None, t[j-1]], 'L'),
            (0, 'T'),
        )

    for i in range(1, m):
        for j in range(1, n):
            V[i][j], M[i][j] = max(
                (V[i-1][j-1] + S[s[i-1], t[j-1]], 'D'),
                (V[i][j-1] + S[None, t[j-1]], 'L'),
                (V[i-1][j] + S[s[i-1], None], 'U'),
                (0, 'T'),
            )

    # reconstruct
    s_idxs = []
    t_idxs = []
    i, j = np.unravel_index(V.argmax(), V.shape)
    bscore = V[i][j]

    while i != 0 or j != 0:
        c = M[i][j]
        if c == 'D':
            i -= 1
            j -= 1
            s_idxs.append(i)
            t_idxs.append(j)
        elif c == 'U':
            i -= 1
        elif c == 'L':
            j -= 1
        else:  # 'T'
            break

    s_idxs.reverse()
    t_idxs.reverse()
    return bscore, s_idxs, t_idxs


# Linear space dynamic programming method


def score_prefix_alignment(s, t, S):
    m = len(s) + 1
    n = len(t) + 1
    V = np.zeros((2, n))

    for j in range(1, n):
        V[0][j] = V[0][j-1] + S[None, t[j-1]]

    for i in range(1, m):
        V[1][0] = V[0][0] + S[s[i-1], None]
        for j in range(1, n):
            V[1][j] = max(
                V[0][j-1] + S[s[i-1], t[j-1]],
                V[0][j] + S[s[i-1], None],
                V[1][j-1] + S[None, t[j-1]],
            )
        V[0, :] = V[1, :]
    return V[0]


def nw_align(s, t, S):
    if len(t) == 1 and len(s) > 1:
        score, (W, Z) = nw_align(t, s, S)
        return score, (Z, W)
    # Try to align ---s--- with t
    b = (float('-inf'), (0, 0))
    for i in range(len(t)):
        u = S[s, t[i]] + sum(S[None, t[j]] for j in range(len(t)) if j != i)
        b = max(b, (u, (0, i)))
    return b


def i_add(Z, i):
    return [z+i for z in Z]


def global_align(X, Y, S):
    Z = []
    W = []
    s = 0
    if len(X) == 0:
        s = sum(S[None, y] for y in Y)
    elif len(Y) == 0:
        s = sum(S[x, None] for x in X)
    elif len(X) == 1 or len(Y) == 1:
        s, (i, j) = nw_align(X, Y, S)
        Z = [i]
        W = [j]
    else:
        xlen = len(X)
        xmid = xlen // 2
        score_l = score_prefix_alignment(X[:xmid], Y, S)
        score_r = score_prefix_alignment(X[xmid:][::-1], Y[::-1], S)[::-1]
        ymid = (score_l + score_r).argmax()

        Z1, W1, s1 = global_align(X[:xmid], Y[:ymid], S)
        Z2, W2, s2 = global_align(X[xmid:], Y[ymid:], S)
        Z = Z1 + i_add(Z2, xmid)
        W = W1 + i_add(W2, ymid)
        s = s1 + s2
    return Z, W, s


def find_local_max(alphabet, scores, s, t):
    m = len(s) + 1
    n = len(t) + 1
    V = [[(0, (0, 0))] * n for _ in range(2)]
    S = make_scoring_dict(alphabet, scores)
    # (score, start, end)
    best = (0, (0, 0), (0, 0))

    for j in range(1, n):
        V[0][j] = max(
            (V[0][j-1][0] + S[None, t[j-1]], V[0][j-1][1]),
            (0, (0, j)),
        )
        best = max(best, (*V[0][j], (0, j)))

    for i in range(1, m):
        V[1][0] = max(
            (V[0][0][0] + S[s[i-1], None], V[0][0][1]),
            (0, (i, 0)),
        )
        best = max(best, (*V[1][0], (i, 0)))

        for j in range(1, n):
            V[1][j] = max(
                (V[0][j-1][0] + S[s[i-1], t[j-1]], V[0][j-1][1]),
                (V[0][j][0]   + S[s[i-1], None],   V[0][j][1]),
                (V[1][j-1][0] + S[None, t[j-1]],   V[1][j-1][1]),
                (0, (i, j)),
            )
            best = max(best, (*V[1][j], (i, j)))
        V[0], V[1] = V[1], V[0]
    return best[1], best[2]


def dynproglin(alphabet, scores, s, t):
    (si, sj), (ei, ej) = find_local_max(alphabet, scores, s, t)
    # If one of the substrings is empty, then just
    # give up now.
    if ei == si or ej == sj:
        return 0, [], []
    S = make_scoring_dict(alphabet, scores)
    Z, W, score = global_align(s[si:ei], t[sj:ej], S)
    return score, i_add(Z, si), i_add(W, sj)


# Heuristic method


def compute_index_table(ktup, s):
    index_table = {}
    for i in range(len(s) - ktup + 1):
        sub = s[i:i+ktup]
        if sub not in index_table:
            index_table[sub] = [i]
        else:
            index_table[sub].append(i)
    return index_table


def find_seeds(alphabet, scores, ktup, index_table, t):
    MIN = (float('+inf'), float('+inf'))
    min_table = {}
    score_table = defaultdict(int)
    S = make_scoring_dict(alphabet, scores)

    for j in range(len(t) - ktup + 1):
        sub = t[j:j+ktup]
        for i in index_table.get(sub, ()):
            d = i - j
            score_table[d] += sum(S[x, x] for x in sub)
            min_table[d] = min(min_table.get(d, MIN), (i, j))

    best_diagonals = sorted(
        score_table.keys(),
        key=score_table.__getitem__,
        reverse=True,
    )[:10]
    return (min_table[d] for d in best_diagonals)


def banded_dp(alphabet, scores, s, t, k):
    m = len(s)
    n = len(t)
    w = 2*k + 1
    if k >= n or w >= n:
        return dynprog(alphabet, scores, s, t)

    S = make_scoring_dict(alphabet, scores)
    L = min(m, n)
    M = [[''] * w for i in range(L+1)]
    V = np.zeros((L+1, w), dtype=int)
    V[:] = float('-inf')

    def get_index(i, j):
        if not 0 <= j <= n or not 0 <= i <= L:
            return None
        t = j - i
        if not -k <= t <= k:
            return None
        center = \
            i if i <= k else \
            w-(n-i+1) if i > n - k else \
            k
        return i, center + t

    def get(i, j):
        u = get_index(i, j)
        if u is not None:
            a, b = u
            return V[a][b]
        return float('-inf')

    def set_max(i, j, v, sym):
        u = get_index(i, j)
        if u is not None:
            a, b = u
            V[a][b], M[a][b] = max((V[a][b], M[a][b]), (v, sym))

    set_max(0, 0, 0, '')
    for i in range(1, m+1):
        set_max(i, 0, get(i-1, 0) + S[s[i-1], None], 'L')
        set_max(i, 0, 0, 'T')

    for j in range(1, k+1):
        set_max(0, j, get(0, j-1) + S[None, t[j-1]], 'U')
        set_max(0, j, 0, 'T')

    for i in range(1, m+1):
        for dj in range(-k, k+1):
            j = i + dj
            set_max(i, j, 0, 'T')
            if 0 < j <= n:
                set_max(i, j, get(i-1, j-1) + S[s[i-1], t[j-1]], 'D')
                set_max(i, j, get(i-1, j) + S[s[i-1], None], 'U')
                set_max(i, j, get(i, j-1) + S[None, t[j-1]], 'L')

    def normalise_j(i, j):
        if i <= k:
            return j
        center = \
            i if i <= k else \
            w-(n-i+1) if i > n - k else \
            k
        return i + j - center

    i, j = np.unravel_index(V.argmax(), V.shape)
    bscore = V[i][j]
    s_idxs = []
    t_idxs = []

    while i != 0 or j != 0:
        c = M[i][j]
        if c == 'D':
            if i <= k or i > n - k:
                j -= 1
            i -= 1
            s_idxs.append(i)
            t_idxs.append(normalise_j(i, j))
        elif c == 'U':
            if i > k and i <= n - k:
                j -= 1
            i -= 1
        elif c == 'L':
            j -= 1
        else:  # 'T'
            break

    s_idxs.reverse()
    t_idxs.reverse()
    return bscore, s_idxs, t_idxs


def heuralign(alphabet, scores, s, t):
    w = 12
    ktup = 2

    it = compute_index_table(ktup, s)
    best = None
    for si, sj in find_seeds(alphabet, scores, ktup, it, t):
        curr = banded_dp(alphabet, scores, s[si:], t[sj:], w)
        if best is None or curr[0] > best[0]:
            best = (curr[0],
                    i_add(curr[1], si),
                    i_add(curr[2], sj))
    return best
