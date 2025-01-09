const math = require('mathjs');

// -----------------------------
// Target and Penalty Functions
// -----------------------------

/**
 * Computes the radius of the sphere based on input millamps.
 * @param {number} milliamps - Input value used to compute the radius.
 * @returns {number} - Radius of the sphere.
 */
const computeRadius = (milliamps) => {
    const radius = (milliamps - 0.1) / 0.22;
    return milliamps > 0.1 ? math.sqrt(radius) : 0;
}

/**
 * Assigns values to the entire landscape based on the sphere's center and radius.
 * Uses vectorized operations with math.js for improved performance.
 * @param {Array} L - Array of points, each row is a point with each column being an [x, y, z, magnitude].
 * @param {Array} center - Center of the sphere [x0, y0, z0].
 * @param {number} r - Radius of the sphere.
 * @returns {Array} - Array of 0s and 1s representing whether each point lies inside the sphere.
 */
const assignSphereValues = (L, center, r) => {
    const [x0, y0, z0] = center;
    // Extract the x, y, z coordinates as separate arrays
    const xCoords = math.column(L, 0);
    const yCoords = math.column(L, 1);
    const zCoords = math.column(L, 2);
    // Compute the squared distance from each point to the sphere center
    const dx = math.subtract(xCoords, x0);
    const dy = math.subtract(yCoords, y0);
    const dz = math.subtract(zCoords, z0);
    const distanceSquared = math.add(
        math.square(dx),
        math.square(dy),
        math.square(dz)
    );
    const radiusSquared = r ** 2; // Faster to square the radius for comparison than to root every squared distance.
    // Perform element-wise comparison to get boolean array and convert to 0/1
    const insideSphere = math.smaller(distanceSquared, radiusSquared); // Gets booleans via inequality
    const sphereMask = math.multiply(insideSphere, 1); // Converts trues to 1s. 
    return sphereMask 
};


/**
 * Calculates the dot product of the sphere assignment vector and the flattened landscape.
 * @param {Array} S - Sphere assignment vector.
 * @param {Array} L - Flattened landscape values array [x,y,z,magnitude]. Same organization of the above rows as points, 
 *                      but with a single column being value at that point.
 *                      Each row should correspond with the row of the x,y,z in landscape for assignSphereValues
 * @returns {number} - Dot product result.
 */
const dotProduct = (S, L) => {
    const magnitudes = math.column(L, 3);
    return math.dot(S, magnitudes);
}

/**
 * Calculates the target function value T(r), which is the density of 'high' values inside a sphere.
 * @param {Array} S_r - Sphere assignment vector.
 * @param {Array} L - Flattened landscape values.
 * @returns {number} - Target function value.
 */
const targetFunction = (S_r, L, r, weight=1) => {
    const numerator = dotProduct(S_r, L)*weight;
    return numerator / 4/3 * 3.14 * r**3; // promote density of large values to avoid excess stim
};

/**
 * Computes the sum of the target function values for an array of possible sphere coordinates.
 * Only includes spheres where the corresponding value in v is non-zero.
 * @param {Array} sphereCoords - Array of sphere centers, each center is an [x, y, z] coordinate.
 * @param {Array} v - Array of contact values (e.g., [q1, q2, q3, q4]).
 * @param {Array} L - Flattened landscape values.
 * @returns {number} - Sum of the target function values for all valid spheres.
 */
const targetFunctionHandler = (sphereCoords, v, L) => {
    let sumTargetValue = 0;
    sphereCoords.forEach((center, index) => {
        if (v[index] > 0.1){ //Saves calculations when radius will be 0. 
            const r = computeRadius(v[index]); // the value at index is our amperage. that is related to radius.
            const S_r = assignSphereValues(L, center, r);
            const targetValue = targetFunction(S_r, L, r);
            sumTargetValue += targetValue;
        }
    });
    return sumTargetValue;
};


/**
 * Calculates the penalty for individual contacts. If contact current above 5mA, penalize.
 * @param {number} v - Contact value (milliamperages)
 * @param {number} lambda - Penalty coefficient.
 * @returns {number} - Penalty value.
 */
const penaltyPerContact = (v, lambda) => lambda * 0.01 * computeRadius(v) // Small penalty to activating contacts. Drives to 0 if minimal gain.


/**
 * Handler function to compute the total penalty for a vector of contact values.
 * @param {Array} v - Array of contact values (e.g., [q1, q2, q3, q4]).
 * @param {number} lambda - Penalty coefficient.
 * @returns {number} - Total penalty value for all contacts.
 */
const penaltyPerContactHandler = (v, lambda) => {
    return v.reduce((sum, value) => sum + penaltyPerContact(value, lambda), 0);
};


/**
 * Calculates the penalty across all  contacts. If total current above 6mA, penalize. 
 * @param {Array} v - Array of contact values (milliamperages)
 * @param {number} lambda - Penalty coefficient.
 * @returns {number} - Penalty value.
 */
const penaltyAllContacts = (v, lambda) => {// blocking function of style (1/(block - val))
    if (math.sum(v) > 0.1) {
        return lambda / (computeRadius(5) - computeRadius(math.sum(v))); //all optimizatoins are in terms of r, not v. 
    }
    else {
        return 0
    }
};


/**
 * Computes the loss function value.
 * @param {Array} sphereCoords - Array of sphere centers, each center is an [x, y, z] coordinate.
 * @param {Array} v - Array of contact values (e.g., [q1, q2, q3, q4]). Organized like sphereCoords
 * @param {Array} L - Flattened landscape values.
 * @param {number} lambda - Penalty coefficient.
 * @returns {number} - Loss function value.
 */
const lossFunction = (sphereCoords, v, L, lambda) => {
    const T = targetFunctionHandler(sphereCoords, v, L);      // Compute the total target value across all relevant spheres
    const P1 = penaltyPerContactHandler(v, lambda);           // Compute the penalty for individual contacts
    const P2 = penaltyAllContacts(v, lambda);                 // Compute the overall penalty for the sum of contact values
    return T - P1 - P2;                                       // Return the loss function value
};

// -----------------------------
// ADAM Optimization Functions
// -----------------------------

/**
 * Initializes Adam optimizer states for each parameter.
 * @param {Array} v - Array of initial contact values (e.g., [q1, q2, q3, q4]).
 * @returns {Object} - Adam state containing moment estimates and step counter.
 */
const initializeAdam = (v) => {
    return {
        m: Array(v.length).fill(0), // First moment vector (mean of gradients)
        v: Array(v.length).fill(0), // Second moment vector (variance of gradients)
        t: 0                        // Time step
    };
};

/**
 * Updates the first moment (momentum) estimate.
 * @param {Array} m - Current first moment estimate.
 * @param {Array} gradient - Clipped gradient vector.
 * @param {number} beta1 - Exponential decay rate for the first moment estimate.
 * @returns {Array} - Updated first moment estimate.
 */
const momentum = (m, gradient, beta1) => 
    m.map((m_i, i) => beta1 * m_i + (1 - beta1) * gradient[i]);

/**
 * Updates the second moment (variance) estimate.
 * @param {Array} v - Current second moment estimate.
 * @param {Array} gradient - Clipped gradient vector.
 * @param {number} beta2 - Exponential decay rate for the second moment estimate.
 * @returns {Array} - Updated second moment estimate.
 */
const variance = (v, gradient, beta2) => 
    v.map((v_i, i) => beta2 * v_i + (1 - beta2) * gradient[i] ** 2);

/**
 * Applies bias correction to the first moment estimate.
 * @param {Array} m - First moment estimate.
 * @param {number} beta1 - Exponential decay rate for the first moment estimate.
 * @param {number} t - Current time step.
 * @returns {Array} - Bias-corrected first moment estimate.
 */
const moment_bias_c = (m, beta1, t) => 
    m.map(m_i => m_i / (1 - Math.pow(beta1, t)));

/**
 * Applies bias correction to the second moment estimate.
 * @param {Array} v - Second moment estimate.
 * @param {number} beta2 - Exponential decay rate for the second moment estimate.
 * @param {number} t - Current time step.
 * @returns {Array} - Bias-corrected second moment estimate.
 */
const var_bias_c = (v, beta2, t) => 
    v.map(v_i => v_i / (1 - Math.pow(beta2, t)));

/**
 * Applies ADAM. Helps prevent grad vanish/explosion, while adapting learning
 * If variance in gradient low, learn fast!
 * If really steep gradient, keep the momentum up!
 * @param {array} gradientVector 
 * @param {array} v 
 * @param {object} adamState 
 * @param {number} alpha 
 * @param {number} beta1 
 * @param {number} beta2 
 * @param {number} epsilon 
 * @returns {array} ADAM-adjusted milliamps array
 */
const adamStep = (gradientVector, v, adamState, alpha = 0.001, beta1 = 0.9, beta2 = 0.999,epsilon = 1e-8,
) => {
    adamState.t += 1;                                                //iterate
    adamState.m = momentum(adamState.m, gradientVector, beta1);     //get new moment
    adamState.v = variance(adamState.v, gradientVector, beta2);     //get new variance
    const mHat = moment_bias_c(adamState.m, beta1, adamState.t);     // correct moment bias (i->0, m->0)
    const vHat = var_bias_c(adamState.v, beta2, adamState.t);        // correct moment bias (i->0, v->0)
    const updatedV = v.map(                                          // apply modified gradient
        (v_i, i) => v_i + alpha * mHat[i] / (Math.sqrt(vHat[i]) + epsilon)
    );
    return { updatedV, adamState };
};

// -------------------------------
// Gradient Optimization Functions
// -------------------------------

/**
 * Computes the difference quotient (numerical derivative) for a given element in the vector v.
 * @param {Array} v - Array of contact values (e.g., [q1, q2, q3, q4]).
 * @param {number} lossCurrent - loss at the current array of v.
 * @param {number} index - Index of the element to compute the derivative for.
 * @param {number} h - Small step size for numerical differentiation.
 * @param {Array} sphereCoords - Array of sphere centers.
 * @param {Array} L - Flattened landscape values.
 * @param {number} lambda - Penalty coefficient.
 * @returns {number} - Numerical derivative at the specified index.
 */
const partialDifferenceQuotient = (v, lossCurrent, index, h, sphereCoords, L, lambda) => {
    const vForward = [...v]; // Create a copy of v 
    vForward[index] += h; // Perturb (step forward) by h only for the variable v at index i
    const lossForward = lossFunction(sphereCoords, vForward, L, lambda);
    return (lossForward - lossCurrent) / h;
};


/**
 * Computes the gradient vector of the loss function across all elements in v.
 * Uses the `partialDifferenceQuotient` function for numerical differentiation.
 * @param {Array} v - Array of contact values (e.g., [q1, q2, q3, q4]).
 * @param {number} h - Small step size for numerical differentiation.
 * @param {Array} sphereCoords - Array of sphere centers.
 * @param {Array} L - Flattened landscape values.
 * @param {number} lambda - Penalty coefficient.
 * @returns {Array} - Gradient vector of the loss function.
 */
const gradientVectorHandler = (v, h, sphereCoords, L, lambda) => {
    const lossCurrent = lossFunction(sphereCoords, v, L, lambda); // Compute the current loss
    let gradientVector = v.map((v_i, index) => // Use `partialDifferenceQuotient` for each element in v to construct the gradient vector
        partialDifferenceQuotient(v, lossCurrent, index, h, sphereCoords, L, lambda)
    );
    gradientVector = clipGradient(gradientVector)
    console.log("Gradient vector: " + gradientVector)
    return gradientVector;
};

/**
 * Performs a single step of gradient ascent to update the contact values.
 * Uses the computed gradient vector to adjust each element of v in the direction of increasing the loss function.
 * @param {Array} gradientVector - The gradient vector of the loss function (numerical derivatives for each element of v).
 * @param {Array} v - Array of contact values (e.g., [q1, q2, q3, q4]).
 * @param {number} alpha - Learning rate for gradient ascent (step size).
 * @returns {Array} - Updated array of contact values after the gradient ascent step.
 */
const gradientAscent = (gradientVector, v, alpha) => {
    // Use element-wise addition with math.js for efficient vector operation
    return math.add(v, math.multiply(gradientVector, alpha));
};

// -----------------------------
// Projection Operator Functions
// -----------------------------

const clipAmps = (v, lower_limit = 0) => {
    return v.map(value => Math.max(value, lower_limit));
}

const clipGradient = (gradientVector, limit = 4) => {
    return gradientVector.map(value => Math.max(-limit, Math.min(value, limit)));
};

/**
 * Projects contact values to satisfy individual and total constraints.
 * @param {Array} v - Array of contact values.
 * @param {number} maxTotal - Maximum total contact value.
 * @returns {Array} - Projected contact values.
 */
const projectConstraints = (v, maxTotal = 5) => {
    const currentSum = math.sum(v);
    console.log("----------------------")
    console.log("Original Sum is :" + currentSum)
    console.log("Original values:" + v)
    console.log("----------------------")

    let final;
    if (currentSum === 0) { // Avoid division by 0
        final = Array(v.length).fill(0);
    }
    else {
        final = v.map(v_i => (v_i / currentSum) * maxTotal)
    }
    console.log("Final Sum is: " + math.sum(final));
    console.log("Final Values: " + final);
    return final; //percentage current scaled to max allowed current 
}

const projectAmpConstrains = (v, maxPer = 5) => {
    return v.map(component => Math.min(component, maxPer));
};

// -----------------------------
// Stop Condition Functions
// -----------------------------

/**
 * Checks L1 norm of the gradient
 * @param {array} gradientVector - array containing the gradient
 * @returns {number} l1Norm
 */
const getL1Norm = (gradientVector) => {
    return math.sum(math.abs(gradientVector));
}

/**
 * Checks for stopping conditions during optimization.
 * @param {Array} currentV - Current contact values.
 * @param {Array} gradientVector - Gradient vector of the loss function.
 * @param {number} l1Tolerance - L1 norm threshold for convergence.
 * @returns {boolean} - True if any stopping condition is met; otherwise false.
 */
const checkStopConditions = (currentV, gradientVector, l1Tolerance) => {
    if (currentV.some(isNaN)) {
        console.error("NaN occurred in: " + currentV);
        return true; // Stop optimization
    }
    if (getL1Norm(gradientVector) < l1Tolerance) {
        console.log(`Convergence detected: L1 norm of gradient < ${l1Tolerance}`);
        return true; // Stop optimization
    }
    return false; // Continue optimization
};

// -----------------------
// Orchestration Functions
// -----------------------

/**
 * Orchestrates the optimization process using gradient ascent.
 * 
 * This optimizer is tuned to operate on values around -1 to 1. 
 * values MUST be standardized before entering. 
 * 
 * Penalty 
 * linear           - implemented. per contact, very small penalty to prevent stim unless at least marginal gain
 * blocking         - implemented. overall limit of 5mA is prevented with infinite loss near 5mA
 * 
 * Target  
 * + vals in sphere - implemented
 * normalize to vol - implemented; prevents gradient explosion
 * 
 * GRADIENT
 * clipping         - implemented
 * ADAM             - implemented
 * Projection       - removed
 * 
 * STOP Rules
 * grad L1 norm     - implemented
 * grad L2 norm     - not implemented
 * max iterations   - implemented
 * convergence      - not implemented
 * plateau dtxn     - not implemented
 * 
 * Validation
 * NANS             - implemented
 * Len check        - implemented
 * Min V check      - implemented
 * Sum V Check      - implemented
 * NonNeg Check     - implemented
 * 
 * FINAL Return
 * projection       - removed; just get the optimizer right. 
 * 
 * NOTES:
 * Potential Logical Error: No safeguard against differing array lengths. 
 * Ensure that sphereCoords and v are always of equal length to prevent unintended behavior.
 * 
 * Performs a maximum of 100 iterations and includes placeholders for additional stopping rules.
 * @param {Array} sphereCoords - Array of sphere centers, each center is an [x, y, z] coordinate.
 * @param {Array} v - Initial guess for the array of contact values (e.g., [q1, q2, q3, q4]).
 * @param {Array} L - Flattened landscape values, an array of (n,m) where n is the points and m is 4 cols (x coord,y coord,z coord,magnitude)
 * @param {number} lambda - Penalty coefficient.
 * @param {number} alpha - Learning rate for gradient ascent (step size).
 * @param {number} h - Small step size for numerical differentiation.
 * @returns {Array} - Optimized array of contact values.
 */
const optimizeSphereValues = (sphereCoords, v, L, lambda=1, alpha=0.01, h=0.001, l1Tolerance=0.05) => {
    validateInputs(sphereCoords, v, L); // throw errors if inputs are incorrect. 
    let currentV = [...v]; // Clone the initial guess for v
    let iteration = 0;
    const adamState = initializeAdam(v);
    while (iteration < 50) { //  allows 100 steps of 0.05mA changes (max of 5mA in total change)
        const gradientVector = gradientVectorHandler(currentV, h, sphereCoords, L, lambda);    // get gradient
        const clippedGradient = clipGradient(gradientVector)
        const result = adamStep(clippedGradient, currentV, adamState, alpha);             // Ascend w/ ADAM
        // currentV= gradientAscent(clippedGradient, currentV, alpha);                    // ascend gradient
        currentV = clipAmps(result.updatedV);        
        checkStopConditions(currentV, gradientVector, l1Tolerance)                        // Check Stop
        iteration += 1;
    }
    console.log(`Optimization completed after ${iteration} iterations.`);
    return currentV //projectConstraints(currentV, maxTotal=5);
};

// -----------------------
//      Helper Code
// -----------------------

/**
 * Validates input parameters for the optimizeSphereValues function.
 * @param {Array} sphereCoords - Array of sphere centers, each center is an [x, y, z] coordinate.
 * @param {Array} v - Array of contact values (e.g., [q1, q2, q3, q4]).
 * @param {Array} L - Flattened landscape values, each row is [x, y, z, magnitude].
 * @throws {Error} - If any validation check fails.
 */
const validateInputs = (sphereCoords, v, L) => {
    // Check that all inputs are arrays
    if (!Array.isArray(sphereCoords)) {
        throw new Error('sphereCoords must be an array.');
    }
    if (!Array.isArray(v)) {
        throw new Error('v must be an array.');
    }
    if (!Array.isArray(L)) {
        throw new Error('L must be an array.');
    }

    // Check that arrays are not empty
    if (sphereCoords.length === 0) {
        throw new Error('sphereCoords array cannot be empty.');
    }
    if (v.length === 0) {
        throw new Error('v array cannot be empty.');
    }
    if (L.length === 0) {
        throw new Error('L array cannot be empty.');
    }

    // Check length consistency
    if (sphereCoords.length !== v.length) {
        throw new Error('sphereCoords and v must have the same length.');
    }

    // Check that at least one v is greater than 0.1
    const hasVOverThreshold = v.some(value => typeof value === 'number' && value > 0.1);
    if (!hasVOverThreshold) {
        throw new Error('At least one contact value in v must be greater than 0.1.');
    }
    // Check if sum of V is over 4
    const overMaxThreshold = v.reduce((sum, value) => sum + value, 0) > 4;
    if (overMaxThreshold) {
        throw new Error('Your total amperage is over 4 mA. Reduce total amperage or face the wrath of unstable gradients.');
    }

    // Validate each sphere coordinate
    sphereCoords.forEach((coord, index) => {
        if (!Array.isArray(coord) || coord.length !== 3) {
            throw new Error(`sphereCoords[${index}] must be an array of three numeric values [x, y, z].`);
        }
        coord.forEach((val, subIndex) => {
            if (typeof val !== 'number' || isNaN(val)) {
                throw new Error(`sphereCoords[${index}][${subIndex}] must be a valid number.`);
            }
        });
    });

    // Validate each contact value
    v.forEach((value, index) => {
        if (typeof value !== 'number' || isNaN(value) || value < 0) {
            throw new Error(`v[${index}] must be a non-negative number.`);
        }
    });

    // Validate each landscape point
    L.forEach((point, index) => {
        if (!Array.isArray(point) || point.length !== 4) {
            throw new Error(`L[${index}] must be an array of four numeric values [x, y, z, magnitude].`);
        }
        point.forEach((val, subIndex) => {
            if (typeof val !== 'number' || isNaN(val)) {
                throw new Error(`L[${index}][${subIndex}] must be a valid number.`);
            }
        });
    });
};

module.exports = {
    optimizeSphereValues,
    lossFunction,
    validateInputs,
    computeRadius,
    assignSphereValues,
    dotProduct,
    targetFunction,
    penaltyPerContact,
    penaltyPerContactHandler,
    penaltyAllContacts,
    partialDifferenceQuotient,
    gradientVectorHandler,
    gradientAscent,
};
