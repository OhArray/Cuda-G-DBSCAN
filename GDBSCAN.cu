#include <algorithm>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

struct Point {
    float x, y;
    bool visited; //initialize to 0
    int clusterId; //initialize to -1
    int type; //initialize to noise 
};

#define NOISE -1
#define BORDER 0
#define CORE 1

__device__ float distance(Point p1, Point p2) {
    return hypotf(p1.x - p2.x, p1.y - p2.y);
}

__global__ void makeGraph1(Point *points, int *numNeighbors, int numPoints, float eps) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < numPoints) {
        for (int i = 0; i < numPoints; ++i) {
            if (i == tid) {
                continue;  // Skip the point itself
            }
            if (distance(points[tid], points[i]) <= eps )
                numNeighbors[tid]++;
        }
    }
}

__global__ void makeGraph2(Point *points, int *adjList, int *startPos, int numPoints, float eps, int minPts) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < numPoints) {
        int start = startPos[tid];
        
        int numNeighbors = 0;
        for (int i = 0; i < numPoints; ++i) {
            if (i == tid) {
                continue;
            }

            if (distance(points[tid], points[i]) <= eps) {
                adjList[start + numNeighbors] = i;
                numNeighbors++;
            }
        }
        if (numNeighbors + 1 > minPts)
            points[tid].type = CORE;
        else
            points[tid].type = NOISE;
    }
}

int* makeGraph(Point *c_points, float eps, int minPts, int *c_numNeighbors, int* c_startPos, int numPoints) {

    int T = 64;
    int B = (numPoints + T - 1)/ T;

    int *c_adjList;

    int *h_numNeighbors = (int*)malloc(numPoints*sizeof(int));
    int *h_startPos = (int*)malloc(numPoints*sizeof(int));

    makeGraph1 <<<T,  B>>> (c_points, c_numNeighbors, numPoints, eps);

    cudaMemcpy(h_numNeighbors, c_numNeighbors, numPoints * sizeof(int), cudaMemcpyDeviceToHost);

    h_startPos[0] = 0;
    for (int i = 1; i < numPoints; i++) {
        h_startPos[i] = h_startPos[i - 1] + h_numNeighbors[i - 1];
    }
    int adjCount = h_startPos[numPoints - 1]  + h_numNeighbors[numPoints - 1];

    cudaMalloc(&c_adjList, adjCount *sizeof(int)); 

    cudaMemcpy(c_startPos, h_startPos, numPoints * sizeof(int), cudaMemcpyHostToDevice);

    makeGraph2 <<<T, B>>> (c_points, c_adjList, c_startPos, numPoints, eps, minPts);

    return c_adjList;
}

__global__ void GPU_BFS_Kernel(int *startPos, int *adjList, int *numNeighbors, bool *Fa, bool *Xa, int numPoints) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < numPoints) {
        if (Fa[tid] == 1) {
            Fa[tid] = 0;
            Xa[tid] = 1;

            int startId = startPos[tid];

            for (int i = 0; i < numNeighbors[tid]; ++i) {
                int nid = adjList[startId + i];
                if(Xa[nid] == 0)
                    Fa[nid] = 1;
            }
        }
    }
}

void CPU_BFS(Point *h_points, int *c_startPos, int *c_adjList, int *c_numNeighbors, bool *h_Xa, bool *c_Xa, bool *h_Fa, bool *c_Fa, int v, int clust, int numPoints) {

    memset(h_Fa, 0, numPoints * sizeof(bool));

    cudaMemset(c_Xa, 0, numPoints*sizeof(bool));

    // Put node v in the frontier
    h_Fa[v] = 1;

    int T = 128;
    int B = (numPoints + T - 1)/ T;

    while (std::any_of(h_Fa, h_Fa + numPoints, thrust::identity<bool>())) {
        cudaMemcpy(c_Fa, h_Fa, numPoints*sizeof(bool), cudaMemcpyHostToDevice);
        GPU_BFS_Kernel <<<T, B>>> (c_startPos, c_adjList, c_numNeighbors, c_Fa, c_Xa, numPoints);
        cudaMemcpy(h_Fa, c_Fa, numPoints*sizeof(bool), cudaMemcpyDeviceToHost);
    }

    cudaMemcpy(h_Xa, c_Xa, numPoints*sizeof(bool), cudaMemcpyDeviceToHost);

    // Label nodes
    for (int n = 0; n < numPoints; ++n) {
        if (h_Xa[n] == 1) {
            h_points[n].clusterId = clust;
            h_points[n].visited = 1;
            if (h_points[n].type != CORE) 
                h_points[n].type = BORDER;
        }
    }
}


void IdentifyClusters(Point *h_points, int *startPos, int *adjList, int *numNeighbors, int numPoints) {
    bool *c_Xa, *c_Fa;

    cudaMalloc(&c_Xa, numPoints*sizeof(bool));
    cudaMalloc(&c_Fa, numPoints*sizeof(bool));

    bool *h_Xa = (bool*)malloc(numPoints*sizeof(bool));
    bool *h_Fa = (bool*)malloc(numPoints*sizeof(bool));

    int clusterId = 0;
    for (int i = 0; i < numPoints; ++i) {
        if (h_points[i].visited == 0 && h_points[i].type == CORE) {
            h_points[i].visited = 1;
            h_points[i].clusterId = clusterId;
            CPU_BFS(h_points, startPos, adjList, numNeighbors, h_Xa, c_Xa, h_Fa, c_Fa, i, clusterId, numPoints);
            clusterId++;
        }
    }

    cudaFree(c_Xa);
    cudaFree(c_Fa);
    free(h_Xa);
    free(h_Fa);
}

// Function to read points from a file
struct Point* readPointsFromFile(const char* filename, int* numPoints) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Error opening the file.\n");
        *numPoints = 0;
        return NULL;
    }

    struct Point* points = NULL;
    float x, y;

    // Count the number of lines in the file
    *numPoints = 0;
    while (fscanf(file, "%f %f", &x, &y) == 2) {
        (*numPoints)++;
    }

    // Reset file position to the beginning
    fseek(file, 0, SEEK_SET);

    // Allocate memory for the points
    points = (struct Point*)malloc(*numPoints * sizeof(struct Point));
    if (points == NULL) {
        fprintf(stderr, "Error allocating memory.\n");
        fclose(file);
        return NULL;
    }

    // Read points from the file
    for (int i = 0; i < *numPoints; ++i) {
        if (fscanf(file, "%f %f", &x, &y) != 2) {
            fprintf(stderr, "Error reading points from the file.\n");
            free(points); // Free allocated memory
            fclose(file);
            return NULL;
        }

        // Assign values to the struct members
        points[i].x = x;
        points[i].y = y;
        points[i].visited = 0;    // false
        points[i].clusterId = -1;
        points[i].type = -1;
    }

    fclose(file);

    return points;
}

void writeClustersToFile(const char* filename, const struct Point* points, int numPoints) {
    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        fprintf(stderr, "Error opening the file for writing.\n");
        return;
    }

    fprintf(file, "x,y,clusterId,type\n");

    for (int i = 0; i < numPoints; ++i) {
        fprintf(file, "%f,%f,%d, %d\n", points[i].x, points[i].y, points[i].clusterId, points[i].type);
    }

    fclose(file);
}

int main(int argc, char *argv[]) {

    if (argc < 4) {
        printf("Usage: %s <eps> <minPts> <filename>\n", argv[0]);
        return 1;
    }

    const char *filename = argv[3];
    const char *outputFilename = "output_clusters.csv";

    float eps = atof(argv[1]); //5
    int minPts = atoi(argv[2]); //1
    int numPoints;

    Point *c_points;
    int *c_numNeighbors, *c_startPos;

    struct Point *h_points = readPointsFromFile(filename, &numPoints);

    cudaMalloc(&c_points, numPoints * sizeof(Point));
    cudaMalloc(&c_startPos, numPoints * sizeof(int));
    cudaMalloc(&c_numNeighbors, numPoints * sizeof(int));

    cudaMemcpy(c_points, h_points, numPoints * sizeof(Point), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Time both makeGraph and IdentifyClusters together
    cudaEventRecord(start);

    int *c_adjList = makeGraph(c_points, eps, minPts, c_numNeighbors, c_startPos, numPoints);
    cudaMemcpy(h_points, c_points, numPoints * sizeof(Point), cudaMemcpyDeviceToHost);
    IdentifyClusters(h_points, c_startPos, c_adjList, c_numNeighbors, numPoints);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds_total = 0;
    cudaEventElapsedTime(&milliseconds_total, start, stop);

    printf("Time taken for makeGraph and IdentifyClusters together: %f milliseconds\n", milliseconds_total);

    writeClustersToFile(outputFilename, h_points, numPoints);

    cudaFree(c_points);
    cudaFree(c_startPos);
    cudaFree(c_numNeighbors);

    return 0;
}
