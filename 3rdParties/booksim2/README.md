# How to compile booksim2 API

```bash
# Compile booksim-api
cd 3rdParties/booksim2/api
mkdir build
cd build
cmake ..
make

# Go back & test
cd ../../../..
python 3rdParties/booksim2/api/booksim_sync.py
```

Example results:

```log
...
Overall average injected packet size = 2.2 (1 samples)
Overall average accepted packet size = 2.2 (1 samples)
Overall average hops = 7.08 (1 samples)
Overall workload runtime = 3871 (1 samples)
Total run time 0.311868
***** Run booksim done *****
```