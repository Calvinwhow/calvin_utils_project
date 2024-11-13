/**
 * Calculates the surface area of a contact.
 *
 * @param {number} diameter - The diameter of the electrode.
 * @param {number} height - The height of the contact.
 * @param {number} n_contacts - The number of contacts in a directional segment.
 * @returns {number} - The surface area of the cylinder.
 */
export const cylinderSurfaceArea = (diameter, height, n_contacts=1) => {
    const radius = diameter / 2;
    const contact_area = 2 * Math.PI * radius * height
    const directional_area = contact_area / n_contacts
    return directional_area
};
