#include <algorithm>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <cuda.h>
#include <math.h>

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
    return  sqrtf(powf(p1.x - p2.x, 2) + powf(p1.y - p2.y, 2));
}

__global__ void makeGraph1(Point *points, int *numNeighbors, int numPoints, int eps) {

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

__global__ void makeGraph2(Point *points, int *adjList, int *startPos, int numPoints, int eps, int minPts) {

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
        if (numNeighbors > minPts)
            points[tid].type = CORE;
        else
            points[tid].type = NOISE;
    }
}

int* makeGraph(Point *c_points, int eps, int minPts, int *c_numNeighbors, int* c_startPos, int numPoints) {

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
    int *h_adjList = (int*)malloc(adjCount*sizeof(int)); //do not need 

    cudaMemcpy(c_startPos, h_startPos, numPoints * sizeof(int), cudaMemcpyHostToDevice);

    makeGraph2 <<<T, B>>> (c_points, c_adjList, c_startPos, numPoints, eps, minPts);

    cudaMemcpy(h_adjList, c_adjList, adjCount * sizeof(int), cudaMemcpyDeviceToHost); //do not need

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

void CPU_BFS(Point *h_points, int *c_startPos, int *c_adjList, int *c_numNeighbors, int v, int clust, int numPoints) {

    bool *c_Xa, *c_Fa;

    cudaMalloc(&c_Xa, numPoints*sizeof(bool));
    cudaMalloc(&c_Fa, numPoints*sizeof(bool));

    bool *h_Xa = (bool*)malloc(numPoints*sizeof(bool));
    bool *h_Fa = (bool*)malloc(numPoints*sizeof(bool));

    memset(h_Xa, 0, numPoints * sizeof(bool));
    memset(h_Fa, 0, numPoints * sizeof(bool));

    cudaMemcpy(c_Xa, h_Xa, numPoints*sizeof(bool), cudaMemcpyHostToDevice);

    // Put node v in the frontier
    h_Fa[v] = 1;

    int T = 64;
    int B = (numPoints + T - 1)/ T;

    // While F has some node with a value of true

    while (std::any_of(h_Fa, h_Fa + numPoints, thrust::identity<bool>())) {
        cudaMemcpy(c_Fa, h_Fa, numPoints*sizeof(bool), cudaMemcpyHostToDevice);
        
        GPU_BFS_Kernel <<<T, B>>> (c_startPos, c_adjList, c_numNeighbors, c_Fa, c_Xa, numPoints);
        cudaDeviceSynchronize();

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

    // Free memory
    cudaFree(c_Xa);
    cudaFree(c_Fa);
    free(h_Xa);
    free(h_Fa);
}


void IdentifyClusters(Point *h_points, int *startPos, int *adjList, int *numNeighbors, int numPoints) {
    int clusterId = 0;
    for (int i = 0; i < numPoints; ++i) {
        if (h_points[i].visited == 0 && h_points[i].type == CORE) {
            h_points[i].visited = 1;
            h_points[i].clusterId = clusterId;
            CPU_BFS(h_points, startPos, adjList, numNeighbors, i, clusterId, numPoints);
            clusterId++;
        }
    }
}

struct Point* readPointsFromFile(const char* filename, int* numPoints) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Error opening the file.\n");
        *numPoints = 0;
        return NULL;
    }

    struct Point* points = NULL;
    float x, y;
    int ignoredValue;

    // Count the number of lines in the file
    *numPoints = 0;
    while (fscanf(file, "%f %f %d", &x, &y, &ignoredValue) == 3) {
        (*numPoints)++;
    }

    // Reset file position to the beginning
    fseek(file, 0, SEEK_SET);

    // Allocate memory for the points
    points = (struct Point*)malloc(*numPoints * sizeof(struct Point));

    // Read points from the file
    for (int i = 0; i < *numPoints; ++i) {
        fscanf(file, "%f %f %d", &x, &y, &ignoredValue);

        points[i].x = x;
        points[i].y = y;
        points[i].visited = 0; // false
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

    const char *filename = "output.txt";
    const char *outputFilename = "output_clusters.csv";

    int eps = atoi(argv[1]); //5
    int minPts = atoi(argv[2]); //1
    int numPoints;

    Point *c_points;
    int *c_numNeighbors, *c_startPos;

    struct Point *h_points = readPointsFromFile(filename, &numPoints);

    cudaMalloc(&c_points, numPoints *sizeof(Point));
    cudaMalloc(&c_startPos, numPoints *sizeof(int));
    cudaMalloc(&c_numNeighbors, numPoints *sizeof(int));

    cudaMemcpy(c_points, h_points, numPoints * sizeof(Point), cudaMemcpyHostToDevice);

    int *c_adjList = makeGraph(c_points, eps, minPts, c_numNeighbors, c_startPos, numPoints); 

    cudaMemcpy(h_points, c_points, numPoints * sizeof(Point), cudaMemcpyDeviceToHost);

    IdentifyClusters(h_points, c_startPos, c_adjList, c_numNeighbors, numPoints);

    writeClustersToFile(outputFilename, h_points, numPoints);

    return 0;
}