/**
 * Takes a charge and converts it to microamps. Can be used to convert a max safe charge into a max safe amperage.
 *  I = Time/Q  <- this is really just an integration of total charge over time. 
 * @param {number} charge - Current in microcoulombs (µC).
 * @param {number} pulseWidth - Pulse width in seconds. In biphasic stim, we want HALF the overall time. 
 * @returns {number} current (I) in microamperes (µA).
 */
export function calculateAmps(charge, pulseWidth) {
    // Convert charge from microcoloumbs (µC) to amps (C) for calculation
    const chargeInCoulombs = charge * 1e-6;
    // Calculate current (I) in amps (A)
    const currentInAmps = pulseWidth / chargeInCoulombs;
    // Convert current to microamps (µA)
    const currentInMicroAmps = currentInAmps * 1e6;
    return currentInMicroAmps;
}
