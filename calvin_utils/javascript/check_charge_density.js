/**
 * Checks if the given charge per phase (Q) is within the safe limit defined by the Shannon equation.
 * Shannon = log(Q/A) = k - log(D)
 * Rearrange for Q
 * Q = A*10^( k - log(D) )
 * @param {number} Q - Charge per phase in microcoulombs (µC).
 * @param {number} A - Electrode surface area in square centimeters (cm²).
 * @param {number} D - Pulse duration in seconds. AKA, pulsewidth. If biphasic, we expect half the pulse width. 
 * @param {number} k - Shannon constant (default is 1.5).
 * @returns {boolean} True if Q is within the safe limit, otherwise false.
 */
export function isSafeCharge(Q, A, D, k = 1.5) {
    // Calculate the maximum safe charge per phase using the Shannon equation
    const maxSafeCharge = A * Math.pow(10, k - Math.log10(D));
    // Check if the given charge per phase is below the safe threshold and return the max safe charge. 
    return Q < maxSafeCharge, maxSafeCharge
}
