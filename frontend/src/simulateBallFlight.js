import * as CONST from './constants';

export const simulateBallFlight = (params) => {
  const { clubheadSpeed, launchAngle, spinRate } = params;

  // Convert inputs to metric units
  const ballSpeed = clubheadSpeed * CONST.SMASH_FACTOR * CONST.MPH_TO_MS;
  const launchRad = (launchAngle * Math.PI) / 180;
  const omega = spinRate * CONST.RPM_TO_RADS;

  let vx = ballSpeed * Math.cos(launchRad);
  let vy = ballSpeed * Math.sin(launchRad);

  const dt = 0.01; // time step
  let x = 0, y = 0, t = 0, apex = 0;
  let trajectory = [];

  while (y >= 0 && t < 10) {
    const v = Math.sqrt(vx ** 2 + vy ** 2);
    
    // Drag force calculation
    const dragForce = 0.5 * CONST.AIR_DENSITY * v ** 2 * CONST.DRAG_COEFF * CONST.BALL_AREA;
    const dragX = -(dragForce / CONST.BALL_MASS) * (vx / v);
    const dragY = -(dragForce / CONST.BALL_MASS) * (vy / v);
    
    // Magnus force calculation
    const magnusForce = 0.5 * CONST.AIR_DENSITY * v * omega * CONST.BALL_AREA * CONST.LIFT_COEFF;
    const liftY = magnusForce / CONST.BALL_MASS;

    // Update velocities and positions
    vx += dragX * dt;
    vy += (dragY + liftY - CONST.G) * dt;
    x += vx * dt;
    y += vy * dt;

    apex = Math.max(apex, y);

    if (trajectory.length % 5 === 0) {
      trajectory.push({
        distance: x * CONST.METERS_TO_YARDS,
        height: y * CONST.METERS_TO_YARDS,
        time: t
      });
    }
    t += dt;
  }

  return {
    trajectory,
    carryYards: Math.round(x * CONST.METERS_TO_YARDS),
    apexYards: Math.round(apex * CONST.METERS_TO_YARDS),
    flightTime: t.toFixed(2)
  };
};