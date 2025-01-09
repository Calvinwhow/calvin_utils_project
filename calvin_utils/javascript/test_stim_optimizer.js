// test_stim_optimizer.js
console.log('Script started');

// Import necessary modules
const math = require('mathjs');
const fs = require('fs');
const csv = require('csv-parser');

const {
    optimizeSphereValues,
    lossFunction,
    // Include other necessary functions if needed
} = require('./stim_optimizer');

// -----------------------------
// Helper Functions for Data Generation
// -----------------------------

/**
 * Converts degrees to radians.
 * @param {number} degrees - Angle in degrees.
 * @returns {number} - Angle in radians.
 */
const degreesToRadians = (degrees) => {
    return degrees * (Math.PI / 180);
};

/**
 * Generates landscape data (L) with x, y, z increasing linearly from 0 to 360.
 * Magnitude m = sin(x) + sin(y) + sin(z), with x, y, z in degrees.
 * @param {number} numPoints - Number of landscape points to generate.
 * @returns {Array} - Array of landscape points, each [x, y, z, m].
 */
const generateLandscapeData = (numPoints) => {
    const L = [];
    for (let i = 0; i < numPoints; i++) {
        const x = i;
        const y = i;
        const z = i;
        const m = Math.sin(degreesToRadians(x)) + Math.sin(degreesToRadians(y)) + Math.sin(degreesToRadians(z));
        L.push([x, y, z, m]);
    }
    return L;
};

/**
 * Generates sphere coordinates positioned at multiples of 90 degrees in each axis.
 * @param {number} numSpheres - Number of spheres to generate.
 * @param {number} centerOffset - Offset from multiples of 90 degrees for sphere centers.
 * @returns {Object} - Object containing sphereCoords and initial contact values (v).
 */
const generateSphereData = (numSpheres, centerOffset = 10) => {
    const sphereCoords = [
        [9.82884, -14.5806, -9.40043], 
        [10.396095120204766, -12.949309229065388, -7.437460523603327], 
        [9.447581079166941, -14.263587072659503, -6.835302019726069], 
        [11.126503800628294, -14.487003698275112, -7.134807456670606], 
        [10.890648453538098, -12.268675895732052, -5.172887190269993], 
        [9.942134412500273, -13.582953739326168, -4.570728686392735], 
        [11.621057133961626, -13.806370364941777, -4.870234123337272], 
        [11.3125, -12.5387, -2.6067099999999996]
    ];
    const v = [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5];

    // for (let i = 0; i < numSpheres; i++) {
    //     // // Position spheres at multiples of 90 degrees with optional offset
    //     // const x = 90 * i + centerOffset;
    //     // const y = 90 * i + centerOffset;
    //     // const z = 90 * i + centerOffset;
    //     // sphereCoords.push([x, y, z]);

    //     // Initialize contact values randomly between 1 and 10
    //     const initialV = Math.random() * 0.1 + 1; // (0, 1)]
    //     v.push(initialV);
    // }

    return { sphereCoords, v };
};

/**
 * Gets the value of L at a specific point by finding the nearest point in L.
 * @param {Array} L - Array of landscape points, each [x, y, z, m].
 * @param {Array} point - The point [x, y, z] to get the L value at.
 * @returns {number} - The L value at the closest point to the given point.
 */
const getLandscapeValueAtPoint = (L, point) => {
    let minDistance = Infinity;
    let closestMValue = null;
    L.forEach((lPoint) => {
        const dx = lPoint[0] - point[0];
        const dy = lPoint[1] - point[1];
        const dz = lPoint[2] - point[2];
        const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (distance < minDistance) {
            minDistance = distance;
            closestMValue = lPoint[3];
        }
    });
    return closestMValue;
};

/**
 * Gets the contacts' coordinates, their voltages, and the value of L at those points.
 * @param {Array} sphereCoords - Array of sphere centers, each [x, y, z].
 * @param {Array} v - Array of contact voltages.
 * @param {Array} L - Array of landscape points, each [x, y, z, m].
 * @returns {Array} - Array of contact information objects.
 */
const getContactsInfo = (sphereCoords, v, L) => {
    const contactsInfo = sphereCoords.map((coord, index) => {
        const voltage = v[index];
        const mValue = getLandscapeValueAtPoint(L, coord);
        return {
            contactIndex: index + 1,
            coordinates: coord,
            voltage: voltage,
            Lvalue: mValue
        };
    });
    return contactsInfo;
};

/**
 * Prints the contacts' information.
 * @param {Array} contactsInfo - Array of contact information objects.
 */
const printContactsInfo = (contactsInfo) => {
    console.log('Contacts Information:');
    // contactsInfo.forEach((info) => {
    //     console.log(`Contact ${info.contactIndex}: Coordinates: [${info.coordinates[0]}, ${info.coordinates[1]}, ${info.coordinates[2]}], Voltage: ${info.voltage.toFixed(4)}, L value at point: ${info.Lvalue.toFixed(4)}`);
    // });
};

// -----------------------------
// Experiment and Grid Search Functions
// -----------------------------

/**
 * Runs an experiment with given hyperparameters and returns the results.
 * @param {Array} L - Landscape data.
 * @param {Array} sphereCoords - Sphere coordinates.
 * @param {Array} initialV - Initial contact values.
 * @param {number} lambda - Penalty coefficient.
 * @param {number} alpha - Learning rate.
 * @param {number} h - Step size for numerical differentiation.
 * @returns {Object} - Results of the experiment.
 */
const runExperiment = (L, sphereCoords, initialV, lambda, alpha, h) => {
    // Perform optimization
    const optimizedV = optimizeSphereValues(sphereCoords, initialV, L, lambda, alpha, h);

    // Compute initial and final loss values
    const initialLoss = lossFunction(sphereCoords, initialV, L, lambda);
    const finalLoss = lossFunction(sphereCoords, optimizedV, L, lambda);

    // Collect contacts info before and after optimization
    const contactsInfoBefore = getContactsInfo(sphereCoords, initialV, L);
    const contactsInfoAfter = getContactsInfo(sphereCoords, optimizedV, L);

    // Return the results
    return {
        optimizedV,
        initialLoss,
        finalLoss,
        contactsInfoBefore,
        contactsInfoAfter
    };
};

/**
 * Synchronously loads landscape data (L) from a CSV file.
 * @param {string} csvPath - Path to the CSV file.
 * @returns {Array} - Array of landscape points, each [x, y, z, m].
 */
const loadLandscapeData = (csvPath) => {
    const data = fs.readFileSync(csvPath, 'utf-8').trim();
    const lines = data.split('\n').filter(line => line.trim() !== '');
    const L = [];

    for (const line of lines) {
        // Remove quotes and split by comma
        const [x, y, z, m] = line.replace(/"/g, '').split(',').map(Number);
        L.push([x, y, z, m]);
    }

    return L;
};

// -----------------------------
// Main Testing Function
// -----------------------------

const main = () => {
    try {
        console.log("StimTestADAM-V3.02")
        // Parameters for data generation
        const numLandscapePoints = 360;
        const numSpheres = 4;
        const centerOffset = 0;

        // Generate a landscape data follow a defined function (L)
        // const L = generateLandscapeData(numLandscapePoints);
        // Generate a landscape data following a nifti (L)
        const csvPath = './data/L_updrs.csv';
        const L = loadLandscapeData(csvPath); 
        console.log(`Generated Landscape Data (L) with ${L.length} points.`);
        console.log('Sample Landscape Points:', L.slice(0, 5));
        // Generate sphere coordinates and initial contact values (v)
        const { sphereCoords, v: initialV } = generateSphereData(numSpheres, centerOffset);

        // Analyze the landscape data (optional)
        // analyzeLandscapeData(L);

        // Print initial contacts information
        console.log('----------------------------------------------------');
        console.log('Initial Contacts Information:');
        const initialContactsInfo = getContactsInfo(sphereCoords, initialV, L);
        printContactsInfo(initialContactsInfo);
        console.log('----------------------------------------------------');

        // Set up grid search ranges
        const lambda_values = [1];
        const alpha_values = [0.01, 0.001];
        const h_values = [0.001, 0.01];

        // Initialize variables to store the best result
        let bestResult = null;
        let bestParams = null;

        // Grid search over lambda, alpha, h
        for (const lambda of lambda_values) {
            for (const alpha of alpha_values) {
                for (const h of h_values) {
                    console.log(`Running experiment with lambda=${lambda}, alpha=${alpha}, h=${h}`);
                    // Run the experiment
                    const result = runExperiment(L, sphereCoords, initialV, lambda, alpha, h);

                    // Print results
                    console.log(`Results for lambda=${lambda}, alpha=${alpha}, h=${h}:`);
                    console.log('Optimized Contact Values (v):', result.optimizedV.map(v_i => v_i.toFixed(4)));
                    console.log(`Initial Loss: ${result.initialLoss.toFixed(4)}`);
                    console.log(`Final Loss: ${result.finalLoss.toFixed(4)}`);
                    console.log('----------------------------------------------------');

                    // Update best result if necessary
                    if (!bestResult || result.finalLoss > bestResult.finalLoss) {
                        bestResult = result;
                        bestParams = { lambda, alpha, h };
                    }
                }
            }
        }

        // After grid search, print the best result
        console.log('========================================');
        console.log('Best Result from Grid Search:');
        console.log(`Parameters: lambda=${bestParams.lambda}, alpha=${bestParams.alpha}, h=${bestParams.h}`);
        console.log('Optimized Contact Values (v):', bestResult.optimizedV.map(v_i => v_i.toFixed(4)));
        console.log(`Initial Loss: ${bestResult.initialLoss.toFixed(4)}`);
        console.log(`Final Loss: ${bestResult.finalLoss.toFixed(4)}`);
        console.log('Contacts Information After Optimization:');
        printContactsInfo(bestResult.contactsInfoAfter);
        console.log('========================================');

    } catch (error) {
        console.error('Error during optimization:', error.message);
    }
};

// Execute the main function
main();
