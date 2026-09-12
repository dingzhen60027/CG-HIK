# Pre-evaluation implementation corrections

1. The first microbenchmark invocation stopped at record assembly because
   `native_iterations` was supplied twice to `dict`. No microbenchmark output
   table or setting selection had been written. The duplicate serialization key
   was removed; the fixed QP bank is reused unchanged. Only the unfinished
   microbenchmark is rerun. No trajectory outcome informed this correction.
