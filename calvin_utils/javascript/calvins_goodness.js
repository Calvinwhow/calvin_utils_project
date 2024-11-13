/**
 * Calculates the normalized goodness score for a given candidate coordinate.
 * The idea is that we have equal weighting of target ROI and avoid ROI
 * Then, we want minimal prox to target ROI and maximal dist to avoid ROI
 * So, goodness = -(dist to target roi) + (dist to avoid roi)
 * example of contact near avoid roi) goodness = -1 + 0.1 =  -0.9
 * example of contact near target roi) goodness = -0.1 + 1 = 0.9
 * Normalize that bad boi and you got a goodness value between [-1, 1]
 * goodness = avoid - target / avoid + target
 * @param {Array} candidate - The candidate coordinate as [x, y, z].
 * @param {Array} target - The target coordinate as [x, y, z].
 * @param {Array} avoidance - The avoidance coordinate as [x, y, z].
 * @returns {number} - The normalized goodness score, ranging from -1 to 1.
 */
export function calvinsGoodness(candidate, target, avoidance) {
    const euclideanDistance = (point1, point2) => {
        const [x1, y1, z1] = point1;
        const [x2, y2, z2] = point2;
        return Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2);
    };

    const distanceToTarget = euclideanDistance(candidate, target);
    const distanceToAvoidance = euclideanDistance(candidate, avoidance);
    const sum = distanceToTarget + distanceToAvoidance;

    // Calculate the normalized goodness score
    return (distanceToAvoidance - distanceToTarget) / sum;
}

/**
 * Finds the index of the coordinate that optimizes the Calvin's goodness function
 * and returns a ranked list of indices from best to worst.
 * 
 * @param {Array} target - The target coordinate as [x, y, z].
 * @param {Array} avoidance - The avoidance coordinate as [x, y, z].
 * @param {Array} candidates - An array of candidate coordinates, each represented as [x, y, z].
 * @returns {Object} - An object containing:
 *   - bestIndex: The index of the candidate with the highest Calvin's goodness score.
 *   - rankedIndices: An array of indices sorted from best to worst.
 */
export function findOptimalCoordinate(target, avoidance, candidates) {
    const scores = candidates.map((candidate, index) => ({
        index,
        score: calvinsGoodness(candidate, target, avoidance)
    }));

    // Sort by score in descending order (best to worst)
    scores.sort((a, b) => b.score - a.score);

    const bestIndex = scores[0].index;
    const rankedIndices = scores.map(item => item.index);

    return { bestIndex, rankedIndices };
}
