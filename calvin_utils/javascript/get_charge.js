/**
 * Calculates the charge per phase (Q) based on the current (I) and pulse width (PW).
 *  Q = I*Time (Pulse width) <- this is really just an integration of total charge over time. 
 * @param {number} current - Current in microamps (µA).
 * @param {number} pulseWidth - Pulse width in seconds. In biphasic stim, we want HALF the overall time. 
 * @returns {number} Charge per phase (Q) in microcoulombs (µC).
 */
export function calculateCharge(current, pulseWidth) {
    // Convert current from microamps (µA) to amps (A) for calculation
    const currentInAmps = current * 1e-6;
    // Calculate charge (Q) in coulombs (C)
    const chargeInCoulombs = currentInAmps * pulseWidth;
    // Convert charge to microcoulombs (µC)
    const chargeInMicroCoulombs = chargeInCoulombs * 1e6;
    return chargeInMicroCoulombs;
}
