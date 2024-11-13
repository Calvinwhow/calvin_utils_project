/**
 * Calculates the amplitude needed to reach a given Euclidean distance.
 * 
 * Radius = root(( Amplitude - 0.1 )/ 0.22) <-similar to sphere's radius, but empirically scaled for VTAs, probably.
 * Radius^2 = (Amplitude - 0.1)/0.22
 * 0.22(Radius^2) - 0.1 = Ampltide
 * K is your scaling parameter, which empirically fudge-factors your estimate
 * K(radius^2) - 0.1 = Amplitude
 * @param {number} distance - The Euclidean distance between two points.
 * @param {number} [k=1] - The scaling factor (default is 1).
 * @returns {number} - The required amplitude. This is returned in the same scale as distance (e.g. mm -> mA)
 */
export function calculateAmplitude(distance, k = 0.22) {   
    return k*(distance ** 2) - 0.1;
};