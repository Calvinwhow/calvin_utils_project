export function convert_fox_to_mni(coordinates, resolution=2) {
    const offset = [45, 63, 36];
    return coordinates.map((value, index) => (value - offset[index]) * resolution);
}