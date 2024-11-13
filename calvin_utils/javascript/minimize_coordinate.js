/**
 * Finds the index of the coordinate that minimizes the Euclidean distance to the target.
 * Uses the 3-variable Pythagorean theorem to calculate the distance.
 * 
 * @param {Array} target - The target coordinate as [x, y, z].
 * @param {Array} coordinates - An array of coordinates, each represented as [x, y, z].
 * @returns {Object} - An object containing:
 *   - index: The index of the coordinate with the minimum Euclidean distance.
 *   - distance: The minimum Euclidean distance.
 */
export function findNearestCoordinate(target, coordinates) {
    const [x1, y1, z1] = target;
    let minIndex = -1;
    let minDistance = Infinity;

    coordinates.forEach(([x2, y2, z2], index) => {
        const distance = Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2);
        if (distance < minDistance) {
            minDistance = distance;
            minIndex = index;
        }
    });

    return { index: minIndex, distance: minDistance };
}