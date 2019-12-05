# Submission includes all of the below

import numpy as np
from collections import defaultdict
from heapq import nlargest


def make_scoring_dict(alphabet, matrix):
    M = {}
    for i, a in enumerate(alphabet):
        M[a, None] = matrix[i][-1]
        M[None, a] = matrix[-1][i]
        for j, b in enumerate(alphabet):
            M[a, b] = matrix[i][j]
    return M


def dynprog(alphabet, scores, s, t):
    S = make_scoring_dict(alphabet, scores)
    m = len(s)
    n = len(t)
    V = np.zeros((m+1, n+1), dtype=int)
    M = np.chararray((m+1, n+1))
    M[:] = ' '
    # Fill in vertical values
    for i in range(1, m+1):
        V[i, 0], M[i, 0] = max(
            (V[i-1, 0] + S[s[i-1], None], 'U'),
            (0, 'T'),
        )

    # Fill in horizontal values
    for j in range(1, n+1):
        V[0, j], M[0, j] = max(
            (V[0, j-1] + S[None, t[j-1]], 'L'),
            (0, 'T'),
        )

    for i in range(1, m+1):
        for j in range(1, n+1):
            cost_d = V[i-1, j-1] + S[s[i-1], t[j-1]]
            cost_u = V[i-1, j]   + S[s[i-1], None]
            cost_l = V[i, j-1]   + S[None, t[j-1]]
            V[i, j], M[i, j] = max(
                (cost_d, 'D'),
                (cost_u, 'U'),
                (cost_l, 'L'),
                (0,      'T'),
            )

    # Reconstruct
    i, j = np.unravel_index(V.argmax(), V.shape)
    max_score = V[i, j]
    s_idxs = []
    t_idxs = []

    while i != 0 or j != 0:
        c = M[i, j]
        if c == b'D':
            i -= 1
            j -= 1
            s_idxs.append(i)
            t_idxs.append(j)
        elif c == b'U': i -= 1
        elif c == b'L': j -= 1
        else: break

    s_idxs.reverse()
    t_idxs.reverse()
    return max_score, s_idxs, t_idxs


# Dynamic programming with O(n) space


def score_global_alignment(S, s, t):
    m = len(s)
    n = len(t)
    V = np.zeros((2, n+1))

    for j in range(1, n+1):
        V[0, j] = V[0, j-1] + S[None, t[j-1]]

    for i in range(1, m+1):
        V[1, 0] = V[0, 0] + S[s[i-1], None]
        for j in range(1, n+1):
            V[1, j] = max(
                V[0, j-1] + S[s[i-1], t[j-1]],
                V[0, j]   + S[s[i-1], None],
                V[1, j-1] + S[None, t[j-1]],
            )
        V[0, :] = V[1, :]
    return V[0]


def nw_align(S, s, t):
    if len(t) == 1 and len(s) > 1:
        score, W, Z = nw_align(S, t, s)
        return score, Z, W
    # Try to align ---s--- with t
    all_score = sum(S[None, y] for y in t)
    score = S[s, None] + all_score
    Z = []
    W = []
    for j in range(len(t)):
        u = S[s, t[j]] + all_score - S[None, t[j]]
        if u > score:
            score = u
            Z = [0]
            W = [j]
    return score, Z, W


def global_align(S, X, Y):
    Z = []
    W = []
    s = 0
    if len(X) == 0:
        s = sum(S[None, y] for y in Y)
    elif len(Y) == 0:
        s = sum(S[x, None] for x in X)
    elif len(X) == 1 or len(Y) == 1:
        return nw_align(S, X, Y)
    else:
        xlen = len(X)
        xmid = xlen // 2
        L = score_global_alignment(S, X[:xmid], Y)
        R = score_global_alignment(S, X[xmid:][::-1], Y[::-1])
        ymid = (L + R[::-1]).argmax()

        s1, Z1, W1 = global_align(S, X[:xmid], Y[:ymid])
        s2, Z2, W2 = global_align(S, X[xmid:], Y[ymid:])

        Z1.extend(z + xmid for z in Z2)
        W1.extend(w + ymid for w in W2)
        Z = Z1
        W = W1
        s = s1 + s2
    return s, Z, W


def find_local_max(S, s, t):
    m = len(s) + 1
    n = len(t) + 1
    V = [[(0, (0, 0))] * n for _ in range(2)]
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
    S = make_scoring_dict(alphabet, scores)
    (si, sj), (ei, ej) = find_local_max(S, s, t)
    # If one of the substrings is empty, then just give up now.
    if ei == si or ej == sj:
        return 0, [], []
    s, Z, W = global_align(S, s[si:ei], t[sj:ej])
    return s, [z + si for z in Z], [w + sj for w in W]


# Heuristic Method (FASTA-lite)


def banded_dp_local(S, k, s, t):
    m = len(s)
    n = len(t)
    M = {}
    V = {}
    for i in range(m + 1):
        for j in range(max(0, i - k), min(i + k, n) + 1):
            V[i, j] = 0
            M[i, j] = ''

    # Fill in vertical values
    for i in range(1, min(k+1, m+1)):
        V[i, 0], M[i, 0] = max(
            (V[i-1, 0] + S[s[i-1], None], 'U'),
            (0, 'T'),
        )

    # Fill in horizontal values
    for j in range(1, min(k+1, n+1)):
        V[0, j], M[0, j] = max(
            (V[0, j-1] + S[None, t[j-1]], 'L'),
            (0, 'T'),
        )

    for i in range(1, m+1):
        for j in range(max(1, i - k), min(i + k, n) + 1):
            cost_d = V[i-1, j-1] + S[s[i-1], t[j-1]]
            cost_u = V.get((i-1, j), float('-inf')) + S[s[i-1], None]
            cost_l = V.get((i, j-1), float('-inf')) + S[None, t[j-1]]
            V[i, j], M[i, j] = max(
                (cost_d, 'D'),
                (cost_u, 'U'),
                (cost_l, 'L'),
                (0, 'T'),
            )

    # Reconstruct
    i, j = max(V, key=V.__getitem__)
    max_score = V[i, j]
    s_idxs = []
    t_idxs = []

    while True:
        c = M[i, j]
        if c == 'D':
            i -= 1
            j -= 1
            s_idxs.append(i)
            t_idxs.append(j)
        elif c == 'U': i -= 1
        elif c == 'L': j -= 1
        else: break

    s_idxs.reverse()
    t_idxs.reverse()
    return max_score, s_idxs, t_idxs


def compute_index_table(ktup, s):
    table = defaultdict(list)
    for i in range(0, len(s) - ktup + 1):
        sub = s[i:i+ktup]
        table[sub].append(i)
    return table


def index_seeds(table, t, ktup):
    index = defaultdict(int)
    for j in range(0, len(t) - ktup + 1):
        sub = t[j:j+ktup]
        for i in table[sub]:
            index[i - j] += 1
    return index


def find_hotspot(index, width):
    means = {}
    for diag in index:
        total = 0
        for i, neighbour in zip(range(-width, width + 1), range(diag - width, diag + width + 1)):
            total += (1 / (1 + abs(i))) * index.get(neighbour, 0)
        means[diag] = total
    return nlargest(5, means, key=means.__getitem__)


def heuralign(alphabet, scores, s, t):
    # TUNABLE PARAMETERS
    k = 15    # band width
    ktup = 2  # ktup

    S = make_scoring_dict(alphabet, scores)
    index_table = compute_index_table(ktup, s)
    best = (0, [], [])

    found = find_hotspot(index_seeds(index_table, t, ktup), k)
    found.append(0)

    for d in found:
        di = 0
        dj = 0
        if d <= 0:
            di = -d
        else:
            dj = d

        score, s_idxs, t_idxs = banded_dp_local(S, k, s[di:], t[dj:])
        s_idxs = [x + di for x in s_idxs]
        t_idxs = [x + dj for x in t_idxs]
        best = max(best, (score, s_idxs, t_idxs))

    return best
