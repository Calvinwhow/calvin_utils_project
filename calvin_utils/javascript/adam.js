
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
