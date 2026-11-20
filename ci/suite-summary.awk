# The suite's summary from Test's table (`ci/full-suite.sh`, `jobs/full-suite.toml`):
# the header names the columns present (`Pass Fail Error Broken Total Time`),
# and the suite's own row has a value in each of them — e.g. "5713 pass of
# 5713", "5690 pass, 23 fail of 5713". Run as `awk -F'|' -f`, on a log whose
# colour codes are removed.
/^Test Summary:/ { n = split($2, names, " ") }
/^TreeGeneralizedHarmonic\.jl +\|/ && n {
    m = split($2, v, " "); s = ""; total = ""
    for (i = 1; i <= m && i <= n; i++) {
        if (names[i] == "Total") total = v[i]
        else if (names[i] != "Time") s = s (s == "" ? "" : ", ") v[i] " " tolower(names[i])
    }
    line = s " of " total
}
END { print line }
