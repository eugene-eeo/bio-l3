import numpy as np
from heapq import nlargest
from operator import itemgetter


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
    score = S[s, None] + sum(S[None, y] for y in t)
    Z = []
    W = []
    for i in range(len(t)):
        u = S[s, t[i]] + sum(S[None, t[j]] for j in range(len(t)) if j != i)
        if u > score:
            score = u
            Z = [0]
            W = [i]
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
    index_table = {}
    for i in range(len(s) - ktup + 1):
        sub = s[i:i+ktup]
        if sub not in index_table:
            index_table[sub] = [i]
        else:
            index_table[sub].append(i)
    return index_table


def find_seeds(ktup, index_table, t):
    for j in range(len(t) - ktup + 1):
        sub = t[j:j+ktup]
        if sub in index_table:
            for i in index_table[sub]:
                yield i, j


def join_seeds(seeds, ktup):
    table = {}
    for i, j in seeds:
        d = i - j
        if d not in table:
            table[d] = {}
        found = False
        for start, ((a, b), num) in table[d].items():
            if i == a and j == b:
                table[d][start] = ((i + ktup, j + ktup), num + 1)
                found = True
                break
        if not found:
            table[d][i, j] = ((i + ktup, j + ktup), 1)

    for diag, match in table.items():
        for start, (end, num) in match.items():
            yield num, diag, start, end


def rescore_runs(runs, S, s, t):
    m = len(s)
    n = len(t)
    # Reevaluate diagonal runs using our scoring matrix
    for num, diag, start, end in runs:
        si, sj = start
        ei, ej = end
        score = sum(S[s[i], t[j]] for i, j in zip(range(si, min(ei, m)), range(sj, min(ej, n))))
        yield score, diag, start, end


def get_diagonal_runs(hotspots, avg_gap_cost):
    # Assume we get input from join_seeds, then they are
    # grouped by i - j.
    curr_d = None
    curr_diag = None
    for num, diag, start, end in hotspots:
        if curr_d != diag:
            if curr_diag is not None:
                yield tuple(curr_diag)
            curr_diag = [num, start, end]
            curr_d = diag
            continue
        # Try to extend the current diagonal.
        # If gap penalty is not too big then OK
        if curr_diag[1] > start:
            a, b = curr_diag[1]
            c, d = end
        else:
            a, b = curr_diag[2]
            c, d = start
        gap_penalty = (abs(a-c) + abs(b-d)) * avg_gap_cost / 2
        if curr_diag[0] + gap_penalty + num >= 0:
            curr_diag[0] += gap_penalty + num
            if curr_diag[1] > start:
                curr_diag[1] = start
            else:
                curr_diag[2] = end
        else:
            # Otherwise this is a new diagonal
            yield tuple(curr_diag)
            curr_diag = [num, start, end]
    # Don't forget the last one we saw
    if curr_diag:
        yield tuple(curr_diag)


def best_path(rescored_runs, avg_gap_cost):
    adj_list = {}
    for run in rescored_runs:
        adj_list[run] = []

    for u in adj_list:
        # Connect uv iff u_end[i] <= v_start[i] and u_end[j] <= v_start[j]
        for v in adj_list:
            if v is u:
                continue
            u_e = u[2]
            v_s = v[1]
            if u_e[0] <= v_s[0] and u_e[1] <= v_s[1]:
                w = (v_s[0]-u_e[0] + v_s[1]-u_e[1]) * avg_gap_cost
                adj_list[u].append((v, w))

    # Just DFS over paths is OK, graph is acyclic
    best_path = []
    best_score = float('-inf')
    Q = [(x[0], [x]) for x in adj_list]
    while Q:
        score, path = Q.pop()
        if score > best_score:
            best_score = score
            best_path = path
        u = path[-1]
        for v, weight in adj_list[u]:
            Q.append((score + weight + v[0], path + [v]))

    return best_path


def heuralign(alphabet, scores, s, t, ktup=2):
    k = 12
    S = make_scoring_dict(alphabet, scores)
    it = compute_index_table(ktup, s)

    avg_gap_cost = sum(S[a, None] for a in alphabet) / len(alphabet)
    max_cost = max(S.values())

    runs = join_seeds(find_seeds(ktup, it, t), ktup)
    runs = rescore_runs(runs, S, s, t)
    runs = get_diagonal_runs(runs, avg_gap_cost)
    runs = nlargest(10, runs, key=itemgetter(0))
    path = best_path(runs, avg_gap_cost / max_cost)

    if not path:
        return banded_dp_local(S, k, s, t)

    max_dist = 0
    for i, (_, (si, sj), _) in enumerate(path):
        for j, (_, (ei, ej), _) in enumerate(path):
            max_dist = max(max_dist, abs((si - sj) - (ei - ej)))

    _, (si, sj), _ = path[0]
    _, _, (ei, ej) = path[-1]

    score, Z, W = banded_dp_local(S, max_dist, s[si:ei], t[sj:ej])
    Z = [z + si for z in Z]
    W = [w + sj for w in W]
    return score, Z, W
