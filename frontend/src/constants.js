// Physical Constants
export const G = 9.81; // gravity m/s^2
export const AIR_DENSITY = 1.225; // kg/m^3
export const BALL_MASS = 0.0459; // kg
export const BALL_DIAMETER = 0.0427; // m
export const BALL_AREA = Math.PI * (BALL_DIAMETER / 2) ** 2;

// Aerodynamic Coefficients
export const DRAG_COEFF = 0.25;
export const LIFT_COEFF = 0.15;

// Equipment Constants
export const SMASH_FACTOR = 1.48;
export const DEFAULT_LOFT = 10.5;

// Conversion Factors
export const MPH_TO_MS = 0.44704;
export const METERS_TO_YARDS = 1.09361;
export const RPM_TO_RADS = (2 * Math.PI) / 60;