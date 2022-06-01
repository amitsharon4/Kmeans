#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>

static int* kmeans(long, long, long , long, double **, double **);
static double evaluateDistance(double *, double *, long);

static double evaluateDistance(double *x, double *y, long d){
    double result;
    int j;
    result = 0;
    for (j = 0; j < d; j++) {
        result += ((x[j]-y[j])*(x[j]-y[j]));
    }
    return result;
}


static int* kmeans(long K, long N, long d, long MAX_ITER, double **observations, double **centroids){
    int *result;
    double **centroidsCalc;
    int *clusterSizes;
    double minDistance;
    double distance;
    double temp;
    int i;
    int j;
    int numOfIterations;
    int cluster;
    int centroidsDidNotChange;
    numOfIterations = 0;
    centroidsDidNotChange = 0;
    centroidsCalc = (double **)malloc(K * sizeof(double *));
    if (centroidsCalc == NULL){
        return NULL;
    }
    result = (int *) calloc(N, sizeof (int));
    if (result == NULL){
        free(centroidsCalc);
        return NULL;
    }
    while(numOfIterations < MAX_ITER && centroidsDidNotChange == 0){
        clusterSizes = (int*) calloc(K, sizeof(int));
        if (clusterSizes == NULL){
            free(centroidsCalc);
            free(result);
            return NULL;
        }
        for (i = 0; i < K; i++) {
            centroidsCalc[i] = (double *) calloc (d, sizeof(double));
            if (centroidsCalc[i] == NULL){
                for(j=0; j < i; j++){
                    free(centroidsCalc[j]);
                }
                free(centroidsCalc);
                free(result);
                return NULL;
            }
        }
        for (i = 0; i < N; i++) {
            minDistance = evaluateDistance(centroids[0], observations[i], d);
            cluster = 0;
            for (j = 1; j < K; j++) {
                distance = evaluateDistance(centroids[j], observations[i], d);
                if (distance < minDistance) {
                    minDistance = distance;
                    cluster = j;
                }
            }
            for (j = 0; j < d; j++) {
                centroidsCalc[cluster][j] = centroidsCalc[cluster][j] + observations[i][j];
            }
            clusterSizes[cluster]++;
            result[i] = cluster;
        }
        centroidsDidNotChange = 1;
        for (i = 0; i < K; i++) {
            for (j = 0; j < d; j++) {
                temp = centroidsCalc[i][j] / clusterSizes[i];
                if(abs(temp - centroids[i][j]) > 0.0001){
                    centroidsDidNotChange = 0;
                }
                centroids[i][j] = temp;
            }
        }
        numOfIterations++;
        for (i = 0; i < K; i++) {
            free(centroidsCalc[i]);
        }
        free(clusterSizes);
    }
    free(centroidsCalc);
    return result;
}

static double* getVectorFromList(PyObject * list){
    Py_ssize_t length = PyList_Size(list);
    PyObject * item;
    double *vector = (double*) calloc (length, sizeof(double));
    if (vector == NULL){
        return NULL;
    }
    int i;
    for (i=0; i<length; i++){
        item = PyList_GetItem(list, i);
        if(!PyNumber_Check(item)){
            free(vector);
            return NULL;
        }
        vector[i] = PyFloat_AsDouble(item);
    }
    return vector;
}

static double** getMatrixFromListofLists(PyObject * list){
    Py_ssize_t length = PyList_Size(list);
    PyObject * item;
    double **matrix = (double**) calloc (length, sizeof(double));
    if (matrix == NULL){
        return NULL;
    }
    double *vector;
    int i,j;
    for (i=0; i<length; i++){
        item = PyList_GetItem(list, i);
        if(!PyList_Check(item)){
            for (j=0; j<length; j++){
                free(matrix[j]);
            }
            free(matrix);
            return NULL;
        }
        vector = getVectorFromList(item);
        if(!vector){
            for (j=0; j<length; j++){
                free(matrix[j]);
            }
            free(matrix);
            return NULL;
        }
        matrix[i] = vector;
    }
    return matrix;
}

static PyObject* kmeansWrapper(PyObject *self, PyObject *args)
{
    PyObject * pyObservations;
    PyObject * pyCentroids;
    PyObject * pyResult;
    int K;
    int N;
    int d;
    int MAX_ITER;
    int i;
    double **observations;
    double **centroids;
    int *result;
    if(!PyArg_ParseTuple(args, "iiiiOO", &K, &N, &d, &MAX_ITER, &pyObservations, &pyCentroids)) {
        return NULL;
    }
    if (!PyList_Check(pyObservations))
        return NULL;
    if (!PyList_Check(pyCentroids))
        return NULL;
    centroids = getMatrixFromListofLists(pyCentroids);
    if (centroids == NULL){
        return NULL;
    }
    observations = getMatrixFromListofLists(pyObservations);
    if (observations == NULL){
        free(centroids);
        return NULL;
    }
    result = kmeans(K, N, d, MAX_ITER, observations, centroids);
    if (result == NULL){
        free(centroids);
        free(observations);
        return NULL;
    }
    pyResult = PyList_New(N);
    if(!pyResult){
        free(centroids);
        free(observations);
        free(result);
        return NULL;
    }
    for (i=0; i< N; i++){
        PyList_SET_ITEM(pyResult, i, Py_BuildValue("i", result[i]));
    }
    for (i = 0; i < PyList_Size(pyCentroids) ; i++){
        free(centroids[i]);
    }
    for (i = 0; i < PyList_Size(pyObservations); i++){
        free(observations[i]);
    }
    free(result);
    free(observations);
    free(centroids);
    return pyResult;
}

static PyMethodDef kmeansMethods[] = {
        {"kmeans",
         (PyCFunction) kmeansWrapper,
         METH_VARARGS,
         PyDoc_STR("Calculates k centroids for N observations, staring with the given centroids")},
        {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "mykmeanspp",
        NULL,
        -1,
        kmeansMethods
};

PyMODINIT_FUNC
PyInit_mykmeanspp(void)
{
    PyObject *m;
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }
    return m;
}