def banded_dp_local(S, k, s, t):
    m = len(s)
    n = len(t)
    M = {}
    V = {}
    for i in range(m + 1):
        for j in range(max(0, i - k), min(i + k + 1, n + 1)):
            V[i, j] = 0
            M[i, j] = ''

    # Fill in vertical values
    for i in range(1, k+1):
        V[i, 0], M[i, 0] = max(
            (V[i-1, 0] + S[s[i-1], None], 'U'),
            (0, 'T'),
        )

    # Fill in horizontal values
    for j in range(1, k+1):
        V[0, j], M[0, j] = max(
            (V[0, j-1] + S[None, t[j-1]], 'L'),
            (0, 'T'),
        )

    for i in range(1, m+1):
        for j in range(max(1, i - k), min(i + k + 1, n + 1)):
            cost_d = V.get((i-1, j-1), float('-inf')) + S[s[i-1], t[j-1]]
            cost_u = V.get((i-1, j),   float('-inf')) + S[s[i-1], None]
            cost_l = V.get((i, j-1),   float('-inf')) + S[None, t[j-1]]
            V[i, j], M[i, j] = max(
                (cost_d, 'D'),
                (cost_u, 'U'),
                (cost_l, 'L'),
                (0, 'T'),
            )

    # Reconstruct
    i, j = max(V, key=V.get)
    max_score = V[i, j]
    s_idxs = []
    t_idxs = []

    while i != 0 or j != 0:
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
