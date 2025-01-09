#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

// Computes the radius based on milliamps.
double computeRadius(double milliamps) {
    return sqrt((milliamps - 1) / -0.22);
}

// Assigns sphere values based on the center and radius.
void assignSphereValues(double *L, double *center, double r, size_t n, int *sphereMask) {
    double x0 = center[0], y0 = center[1], z0 = center[2];
    double rSquared = r * r;

    for (size_t i = 0; i < n; ++i) {
        double dx = L[i * 3] - x0;
        double dy = L[i * 3 + 1] - y0;
        double dz = L[i * 3 + 2] - z0;
        double distanceSquared = dx * dx + dy * dy + dz * dz;

        sphereMask[i] = (distanceSquared < rSquared) ? 1 : 0;
    }
}

// Computes the dot product of two arrays.
double dotProduct(const int *S, const double *L, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sum += S[i] * L[i];
    }
    return sum;
}

// Computes the target function value.
double targetFunction(double r, int *S_r, double *L, size_t n) {
    double volume = (4.0 * M_PI / 3.0) * r * r * r;
    double value = dotProduct(S_r, L, n);
    return value / volume;
}

// Computes the penalty for individual contacts.
double penaltyPerContact(double v, double lambda) {
    return lambda * fmax(v - 5.0, 0.0);
}

// Computes the penalty for all contacts.
double penaltyAllContacts(double *v, size_t len, double lambda) {
    double total = 0.0;
    for (size_t i = 0; i < len; ++i) {
        total += v[i];
    }
    return lambda * fmax(total - 6.0, 0.0);
}

// Computes the loss function value.
double lossFunction(double *sphereCoords, double *v, double *L, size_t numSpheres, size_t numPoints, double lambda) {
    double totalTargetValue = 0.0;
    for (size_t i = 0; i < numSpheres; ++i) {
        if (v[i] != 0.0) {
            double r = computeRadius(v[i]);
            int *S_r = (int *)malloc(numPoints * sizeof(int));
            assignSphereValues(L, &sphereCoords[i * 3], r, numPoints, S_r);
            totalTargetValue += targetFunction(r, S_r, L, numPoints);
            free(S_r);
        }
    }
    double penalty1 = 0.0;
    for (size_t i = 0; i < numSpheres; ++i) {
        penalty1 += penaltyPerContact(v[i], lambda);
    }
    double penalty2 = penaltyAllContacts(v, numSpheres, lambda);

    return totalTargetValue - penalty1 - penalty2;
}

// Computes numerical gradient vector.
void gradientVector(double *v, double *gradient, double h, double *sphereCoords, double *L, size_t numSpheres, size_t numPoints, double lambda) {
    double lossCurrent = lossFunction(sphereCoords, v, L, numSpheres, numPoints, lambda);

    for (size_t i = 0; i < numSpheres; ++i) {
        double vForward = v[i] + h;
        double original = v[i];
        v[i] = vForward;
        double lossForward = lossFunction(sphereCoords, v, L, numSpheres, numPoints, lambda);
        gradient[i] = (lossForward - lossCurrent) / h;
        v[i] = original;  // Restore original value
    }
}

// Performs a single step of gradient ascent.
void gradientAscent(double *v, double *gradient, double alpha, size_t len) {
    for (size_t i = 0; i < len; ++i) {
        v[i] += alpha * gradient[i];
    }
}

// Optimizes the contact values using gradient ascent.
void optimizeSphereValues(double *sphereCoords, double *v, double *L, size_t numSpheres, size_t numPoints, double lambda, double alpha, double h, int maxIter) {
    double *gradient = (double *)malloc(numSpheres * sizeof(double));

    for (int iter = 0; iter < maxIter; ++iter) {
        gradientVector(v, gradient, h, sphereCoords, L, numSpheres, numPoints, lambda);
        gradientAscent(v, gradient, alpha, numSpheres);
    }

    free(gradient);
}
