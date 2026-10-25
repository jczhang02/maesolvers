# Five levels of reproduction

A reproduction is judged against the source paper's own table, on the paper's own test problem, and the verdict is one of five levels:

1. **Exact**: at every printed $N$, our value agrees with the published one to the printed digits.
2. **Order of magnitude**: the ratio ours / published lies in $[1/2, 2]$ at every printed $N$, and the observed convergence orders differ by at most $0.2$.
3. **Statistical** (neural methods): the published single-run number lies inside our $5$% to $95$% band over $10$ seeds.
4. **Figure only**: the paper prints no table, so the comparison is against a figure.
5. **Mismatch**: none of the above; recorded as a finding in `benchmarks/<method>.md`, never tuned away.

This replaces the single rule "matches to the printed digits", which would have rejected every neural method and every paper whose setting is underspecified.
The thresholds in levels 2 and 3 are our own choice, with no literature basis; they are fixed here so that verdicts from different methods mean the same thing.
`tests/test_reproduction.py` pins the published values of exact-level reproductions; a pinned value changes only together with the finding that justifies it.
