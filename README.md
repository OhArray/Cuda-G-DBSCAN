**Getting It Running**

In order to run this code, I would recommend using CUDA Toolkit Version $\ge$ 12.0.0. I tried to run the code on the Pascal cluster, but it did not work. I would guess that this was due to some versioning discrepancy. To run the code, you need to use .txt files for your points with pairs of coordinates $x$ and $y$. Separate $x$ and $y$ with a space, and each point is on a new line. To compile and run, this is all you need:

```bash
nvcc -o GDBSCAN GDBSCAN.cu
./GDBSCAN <eps> <minPts> <filename>
```
Additionally, as my project proposal required, I created Python code that can take different store names and use the Overpass API from OSM to find the coordinates of businesses with matching names. With this, you can use the output file from that Python script and do cluster analysis on it using customized CUDA code that specifically takes the outputs from the Python script. You can then run it in the Python cluster visualizer that I made. To compile the GDBSCAN_OSM code use:

```bash
nvcc -o GDBSCAN GDBSCAN_OSM.cu
./GDBSCAN <eps> <minPts>
```
