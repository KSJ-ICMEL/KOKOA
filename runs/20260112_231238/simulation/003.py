"""
KOKOA Simulation #3
Generated: 2026-01-12 23:15:37
"""
import os, sys, traceback

# Project root (pre-calculated by Simulator)
_PROJECT_ROOT = "C:/Users/sjkim/KOKOA"
_CIF_PATH = "C:/Users/sjkim/KOKOA/Li4.47La3Zr2O12.cif"

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    os.chdir('C:/Users/sjkim/KOKOA/runs/20260112_231238')
except Exception as e:
    sys.stderr.write(f"Directory Error: {e}\n")

try:
    **Solution Explanation**

    For every test case we are given

    * `n` – number of students  
    * `m` – number of friendships (undirected edges)  
    * `k` – number of students that are already infected  
    * `t` – time limit (in seconds)

    The friendship graph is connected.  
    At time `0` the `k` infected students are the only ones that have the
    virus.  
    During one second a student can infect **at most one** of his friends.
    The infection spreads simultaneously – in one second every infected
    student may infect one of his neighbours.

    For every test case we have to output the maximum possible number of
    students that can be infected after at most `t` seconds.



    --------------------------------------------------------------------

    #### 1.  Observations

    * The infection can be modelled as a **breadth‑first search** (BFS)
      starting from the initially infected vertices.
    * In one second the infection can move one edge away from the already
      infected set.  
      Therefore after `d` seconds the infection can reach all vertices
      whose shortest distance from the set of initially infected vertices
      is at most `d`.
    * The graph is connected, so every vertex is reachable.

    So the answer is simply


    number of vertices whose distance to the set of initially infected
    vertices is ≤ t


    --------------------------------------------------------------------

    #### 2.  Algorithm

    For each test case

    1. Build the adjacency list of the graph.
    2. Initialise a queue with all initially infected vertices and set
       their distance to `0`.
    3. Run a multi‑source BFS:
       * pop a vertex `v` from the queue,
       * for every neighbour `u` of `v` that has not been visited yet  
         set `dist[u] = dist[v] + 1` and push `u` into the queue.
    4. After the BFS, count all vertices with `dist[v] ≤ t`
       (vertices that were never reached have distance `∞`).

    The count is the required maximum number of infected students.

    --------------------------------------------------------------------

    #### 3.  Correctness Proof  

    We prove that the algorithm outputs the maximum possible number of
    infected students.

    ---

    ##### Lemma 1  
    During the infection process a student can become infected no earlier
    than at the moment equal to his shortest distance (in edges) from the
    set of initially infected students.

    **Proof.**

    The infection can only travel along edges, one edge per second.
    Therefore to reach a student at distance `d` at least `d` seconds are
    necessary. ∎



    ##### Lemma 2  
    For every student whose shortest distance from the initially infected
    set is `d ≤ t`, there exists a strategy that infects this student
    within `t` seconds.

    **Proof.**

    Take a shortest path of length `d` from an initially infected student
    to the target student.  
    In the first second infect the neighbour on this path, in the second
    second infect the next one, and so on.  
    After `d` seconds the target student is infected.  
    Because `d ≤ t`, this strategy finishes within the allowed time. ∎



    ##### Lemma 3  
    The BFS performed by the algorithm computes for every vertex its
    shortest distance to the set of initially infected vertices.

    **Proof.**

    The BFS starts simultaneously from all initially infected vertices
    with distance `0`.  
    When a vertex is first taken from the queue, all paths that reach it
    through already processed vertices have length at least the current
    distance.  
    Thus the first time a vertex is discovered is via a shortest path,
    and its stored distance is minimal. ∎



    ##### Lemma 4  
    The algorithm counts exactly all students that can be infected within
    `t` seconds.

    **Proof.**

    By Lemma&nbsp;3 the algorithm obtains the true shortest distance `d`
    for every student.  
    If `d ≤ t`, by Lemma&nbsp;2 the student can be infected within the
    time limit, so it must be counted.  
    If `d > t` or the student is unreachable (distance `∞`), no strategy
    can infect him within `t` seconds (Lemma&nbsp;1), so it must not be
    counted.  
    Therefore the counted set equals the set of all students that can be
    infected within the time limit. ∎



    ##### Theorem  
    For each test case the algorithm outputs the maximum possible number
    of students that can be infected after at most `t` seconds.

    **Proof.**

    By Lemma&nbsp;4 the algorithm counts exactly the students that can be
    infected within the time limit.  
    No strategy can infect more students than this set, and a strategy
    exists that infects all of them (Lemma&nbsp;2).  
    Hence the algorithm’s output is the maximum achievable number. ∎



    --------------------------------------------------------------------

    #### 4.  Complexity Analysis

    Let `n` be the number of students and `m` the number of friendships.

    * Building the adjacency list: `O(n + m)`
    * BFS traversal: each vertex and each edge is processed once → `O(n + m)`
    * Counting the answer: `O(n)`

    Total time per test case: `O(n + m)`  
    Memory consumption: adjacency list `O(n + m)` and distance array `O(n)`.



    --------------------------------------------------------------------

    #### 5.  Reference Implementation  (Python 3)


    import sys
    from collections import deque

    def solve() -> None:
        data = sys.stdin.read().strip().split()
        if not data:
            return
        it = iter(data)
        t_cases = int(next(it))
        out_lines = []

        for _ in range(t_cases):
            n = int(next(it))
            m = int(next(it))
            k = int(next(it))
            t_limit = int(next(it))

            # adjacency list
            adj = [[] for _ in range(n + 1)]
            for _ in range(m):
                u = int(next(it))
                v = int(next(it))
                adj[u].append(v)
                adj[v].append(u)

            # initial infected vertices
            infected = [int(next(it)) for _ in range(k)]

            # multi‑source BFS
            dist = [-1] * (n + 1)
            q = deque()
            for v in infected:
                dist[v] = 0
                q.append(v)

            while q:
                v = q.popleft()
                for nb in adj[v]:
                    if dist[nb] == -1:
                        dist[nb] = dist[v] + 1
                        q.append(nb)

            # count vertices reachable within t_limit seconds
            infected_count = sum(1 for d in dist[1:] if d != -1 and d <= t_limit)
            out_lines.append(str(infected_count))

        sys.stdout.write("\n".join(out_lines))

    if __name__ == "__main__":
        solve()


    The program follows exactly the algorithm proven correct above and
    conforms to the required input and output format.
except Exception as e:
    sys.stderr.write(f"Runtime Error: {str(e)}\n")
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
